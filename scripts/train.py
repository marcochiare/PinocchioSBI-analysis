import argparse
import contextlib
import io
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import warnings

from datetime import datetime

from sbi.diagnostics import run_sbc, run_tarp
from sbi.inference import NPE, EnsemblePosterior
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform

from pinsbi.diagnostics import compute_covariance_posterior, compute_coverage_generic
from pinsbi.neuralnets import FCEmbedding, MatrixEmbedding, MatrixEmbeddingSimple
from pinsbi.utils import load_dataset, select_pars, tarp_bootstrap

#defaults
_DEF_PROBE = 'counts_lmb_z'
_DEF_VALFR = 0.05
_DEF_TESTFR = 0.1
_DEF_N_ENSEMBLE = 5
_DEF_DENS_EST = 'nsf'
_DEF_NUM_TRANSF = 5
_DEF_HIDDENFEAT = 96
_DEF_WHICH_EMBED = 'CNN'
_DEF_CNN_HIDDENFEAT = 128
_DEF_FC_HIDDENFEAT = 64
_DEF_FC_OUTDIM = 8
_DEF_FC_LAYERS = 2
_DEF_BATCHSIZE = 64
_DEF_LR = 1e-4
_DEF_WHICH = ['Om0', 'sigma8', 'h']

_EARLY_STOP = 60

# Prior limits
# Std Sobol sequence
bounds = {
    "Om0":    (0.1, 0.5),
    "sigma8": (0.60, 1.20),
    "h":      (0.60, 0.80),
}
# Dense Sobol sequence
#bounds = {
#        "Om0":    (0.29, 0.33),
#        "sigma8": (0.77, 0.79),
#        "h":      (0.68, 0.72),
#        }

# sbi moves the data from cpu->cuda independently, I simply suppress this unnecessary warning
warnings.filterwarnings("ignore", message=".*Moving.*to the data_device.*", category=UserWarning)
# suppress an annoying userwarning when using a nsf
warnings.filterwarnings("ignore", message=r".*torch\.triangular_solve is deprecated.*", category=UserWarning)

def printlog(text: str, flush: bool = False):
    now = datetime.now().strftime('%H:%M:%S')
    logger = 'TRAIN'
    print(f'\033[92m[{logger:<5}: {now}]>\033[0m ' + text, flush=flush)

def trainable_params(net):
    """
    Counts the number of trainable parameter for a network
    """
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

if __name__ == '__main__':

    # ======================================================================================== #
    # Args
    # ======================================================================================== #

    parser = argparse.ArgumentParser(description='Train a NPE on the number counts')

    parser.add_argument('--parameters-path', type=str, required=True)
    parser.add_argument('--sumstats-path', type=str, required=True)
    parser.add_argument('--weights-path', type=str, required=True)
    parser.add_argument('--extra-tag', type=str, required=False, default=None)

    parser.add_argument('--probe', type=str, default=_DEF_PROBE)
    parser.add_argument('--normalise', action='store_true')
    parser.add_argument('--val-frac', type=float, default=_DEF_VALFR)
    parser.add_argument('--test-frac', type=float, default=_DEF_TESTFR)
    parser.add_argument('--which-param', type=str, nargs='+', default=_DEF_WHICH)
    parser.add_argument('--do-pca', action='store_true')
    parser.add_argument('--ensemble-training', action='store_true')
    parser.add_argument('--num-ensemble', type=int, default=_DEF_N_ENSEMBLE)

    parser.add_argument('--use-embedding', action='store_true')
    parser.add_argument('--which-embedding', type=str, default=_DEF_WHICH_EMBED)
    parser.add_argument('--cnn-hidden-features', type=int, default=_DEF_CNN_HIDDENFEAT) # CNN only
    parser.add_argument('--fc-hidden-features', type=int, default=_DEF_FC_HIDDENFEAT)   # CNN & FC
    parser.add_argument('--fc-out-dimension', type=int, default=_DEF_FC_OUTDIM)         # CNN & FC 
    parser.add_argument('--fc-layers', type=int, default=_DEF_FC_LAYERS)                # FC only
    parser.add_argument('--fc-halve', action='store_true')                              # FC only

    parser.add_argument('--density-est', type=str, default=_DEF_DENS_EST)
    parser.add_argument('--num-transforms', type=int, default=_DEF_NUM_TRANSF)
    parser.add_argument('--hidden-features', type=int, default=_DEF_HIDDENFEAT)
    parser.add_argument('--batchsize', type=int, default=_DEF_BATCHSIZE)
    parser.add_argument('--learning-rate', type=float, default=_DEF_LR)

    args = parser.parse_args()

    if args.do_pca and args.use_embedding:
        raise AttributeError('Cannot use both PCA and an embedding network')

    if args.which_embedding.upper() in ['FC', 'ID']:
        args.flatten = True
    else:
        args.flatten = False

    # ======================================================================================== #

    printlog(f'Found boundaries: {str(bounds)}')
    printlog('== Parsed values ==')
    for arg in vars(args):
        printlog(f'{arg:<20} {getattr(args, arg)}')

    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')
    printlog(f'Found device {device}', flush=True)

    # ======================================================================================== #
    # Load dataset
    # ======================================================================================== #
    
    dat, SHAPE = load_dataset(args.parameters_path, args.sumstats_path, args.probe, normalise=args.normalise,
                              bounds=bounds, return_shape=True, flatten=args.flatten)
    if args.do_pca:
        dat.pca(0.99)
    if args.val_frac > 0.:
        train, val, test = dat.split3(args.val_frac, args.test_frac)
    else:
        train, test = dat.split(args.test_frac)

    # ======================================================================================== #
    # Train
    # ======================================================================================== #

    which_idx, lowb, highb = select_pars(bounds, args.which_param, normalise=args.normalise)
    prior = BoxUniform(low=lowb, high=highb, device=device)

    N_POSTERIORS = args.num_ensemble if args.ensemble_training else 1
    posteriors = []
    inferences = []

    for i in range(N_POSTERIORS):

        if args.use_embedding and args.which_embedding.upper() == 'CNN':
            embedding_net = MatrixEmbeddingSimple(
                    in_shape = SHAPE,
                    output_dim = args.fc_out_dimension,
                    cnn_init_hidden_chn = args.cnn_hidden_features,
                    fc_hidden_chn = args.fc_hidden_features,
                    )
        elif args.use_embedding and args.which_embedding.upper() == 'FC':
            embedding_net = FCEmbedding(
                    in_channels = SHAPE[0],
                    out_channels = args.fc_out_dimension,
                    hidden_channels = args.fc_hidden_features,
                    n_layers = args.fc_layers,
                    halve = args.fc_halve
                    )
        else:
            embedding_net = nn.Identity()
        
        density_estimator = posterior_nn(
                model = args.density_est,
                num_transforms = args.num_transforms,
                hidden_features = args.hidden_features,
                embedding_net = embedding_net,
                )

        inference = NPE(prior = prior, density_estimator = density_estimator, device = device,
                        tracker = None, show_progress_bars = False)
        inference.append_simulations(train.theta[:,which_idx], train.x_pca if args.do_pca else train.x)
        
        # catch the stdout
        captured = io.StringIO()
        init_time = datetime.now()
        printlog(f'Starting the training... [{i+1}/{N_POSTERIORS}]', flush=True)
        with contextlib.redirect_stdout(captured):
            estimator = inference.train(
                    training_batch_size = args.batchsize,
                    learning_rate = args.learning_rate,
                    show_train_summary = False,
                    stop_after_epochs = _EARLY_STOP,
                    )
        sbi_output = captured.getvalue().strip()
        if sbi_output:  printlog(str(sbi_output), flush=True)
        if i == 0:
            printlog(f'Embed. trained parameters:   {trainable_params(embedding_net)}', flush=True)
            printlog(f'NPE trained parameters:      {trainable_params(estimator)}', flush=True)
        printlog(f'Elapsed time: {datetime.now() - init_time}', flush=True)

        posteriors.append(inference.build_posterior(estimator))
        inferences.append(inference)
    
    ensemble_posterior = EnsemblePosterior(posteriors)

    # ======================================================================================== #
    # Save weights
    # ======================================================================================== #

    for i in range(N_POSTERIORS):

        savename = f'{args.probe}_{args.density_est}_n{args.num_transforms}_hidd{args.hidden_features}'
        if args.use_embedding and args.which_embedding.upper() == 'CNN':
            savename += f'_CNN_{args.cnn_hidden_features}-{args.fc_hidden_features}-{args.fc_out_dimension}'
        elif args.use_embedding and args.which_embedding.upper() == 'FC':
            savename += f'_FC_{args.fc_layers}-{args.fc_hidden_features}-{args.fc_out_dimension}-{int(args.fc_halve)}'
        else:
            savename += '_ID'
        if args.do_pca:
            savename += '_PCA'
        if args.normalise:
            savename += '_NORM'
        if args.extra_tag is not None:
            savename += f'_{args.extra_tag}'

        # For the upcoming versions of sbi
        #fullpath = f'{args.weights_path}/inference/{savename}_{i}.pt'
        #inferences[i].save(fullpath)

        #fullpath = f'{args.weights_path}/posteriors/{savename}_{i}.pt'
        #posteriors[i].save(fullpath)

        #printlog(f'Weights saved: {fullpath}', flush=True)

        with open(f"{args.weights_path}/{savename}_{i}.pkl", "wb") as handle:
            pickle.dump(inferences[i], handle)
            printlog(f'Weights saved {args.weights_path}/{savename}_{i}.pkl')

    # ======================================================================================== #
    # Run the diagnostics
    # ======================================================================================== #
    
    test.to('cpu')
    ensemble_posterior.to('cpu')
    cpus = int(os.getenv('SLURM_CPUS_PER_TASK', 1))

    printlog('Running NLP on test', flush=True)
    lp = []
    for th, x in zip(test.theta[:, which_idx], test.x):
        lp.append(ensemble_posterior.log_prob(th[None], x=x[None]).item())
    printlog(f'mean NLP: {-np.mean(lp)}')

    printlog('Running SBC', flush=True)
    num_posterior_samples = 1000
    ranks, dap_samples = run_sbc(
            test.theta[:, which_idx],
            test.x if not args.do_pca else test.x_pca,
            ensemble_posterior,
            num_posterior_samples=num_posterior_samples,
            use_batched_sampling=False,
            num_workers=cpus,
            )

    printlog('Running TARP (bootstrap)', flush=True)
    ecps, alphas = tarp_bootstrap(
            test.theta[:,which_idx],
            test.x if not args.do_pca else test.x_pca,
            ensemble_posterior,
            n_bootstraps=30,
            desc='bootstrap TARP',
            bins=10,
            num_workers=cpus,
            )

    printlog('Running coverage diagnostics', flush=True)
    def my_sampler(nsamples, obs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return ensemble_posterior.sample((nsamples,), x=obs, show_progress_bars=False)
    alphas2, marg, joint = compute_coverage_generic(
            test,
            my_sampler,
            which_idx,
            num_samples=1000,
            n_alpha=20,
            num_workers=cpus,
            )

    printlog('Computing correlation matrix on the test dataset', flush=True)
    cov, cov_std, corr, corr_std = compute_covariance_posterior(
            test,
            ensemble_posterior,
            args.which_param,
            bounds,
            normalise=args.normalise,
            num_workers=cpus,
            )

    save_data = {'ranks': ranks, 'alphas': alphas, 'ecps': ecps,
                 'alphas2': alphas2, 'marg': marg, 'joint': joint,
                 'cov_mean': cov, 'cov_std': cov_std, 'corr_mean': corr, 'corr_std': corr_std,
                 'test_nlp': -np.mean(lp),
                 }
    fullpath = f'{args.weights_path}/{savename}_diagn.npz'
    np.savez(fullpath, **save_data)
    printlog(f'Saved diagnostics: {fullpath}', flush=True)

