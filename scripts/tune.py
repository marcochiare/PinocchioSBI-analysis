"""
This script is used to tune the NPE hyperparameters with optuna

IMPORTANT NOTE
    When k-cross validation is not executed:
        In this case the dataset is loaded and split in 3 (train, val, test) with the first two used for hyperparameter
        tuning and `test` remains untouched. When tuning with this setting, remember to test using this 3-piece split on
        the full dataset (assuming you use the same seed), to prevent training/validation sims to end in test dataset.
    When k-cross validation is executed:
        In this case the dataset is loaded and split in 2 (train, test) which are NOT the same train/test of the above 2-piece
        split. This means that you need to do the same split when loading the dataset for testing. So keep everything consistent!
        Since in this mode we are executing cross-validation, the `train` dataset is used for both training and validation.

IMPORTANT NOTE
    Currently, the dictionary with the low/high boundaries for the cosmological parameters is hardcoded at the beginning of this script.
    A better solution will be found in the future. Cheers.
"""
import argparse
import contextlib
import io
import numpy as np
import os
import optuna
import torch
import torch.nn as nn
import warnings

from datetime import datetime
from functools import partial

from pinsbi.utils import select_pars, load_dataset
from sbi.neural_nets import posterior_nn
from sbi.inference import NPE, EnsemblePosterior
from sbi.utils import BoxUniform

#defaults
_DEF_N_ENSEMBLE = 5
_DEF_VALFR = 0.1
_DEF_TESTFR = 0.1
_DEF_DENS_EST = ['maf']
_DEF_NUM_TRANSF_MAX = 10
_DEF_HIDDENFEAT = [64, 128]
_DEF_CNN_HIDDENFEAT = [16, 32, 64]
_DEF_FC_HIDDENFEAT = [64, 128]
_DEF_FC_OUTDIM = [8, 16, 32]
_DEF_FC_LAYERS_MAX = 5
_DEF_BATCHSIZE = [128]
_DEF_LR_MIN = 1e-6
_DEF_LR_MAX = 1e-3
_DEF_NTRIALS = 100
_DEF_NTRIALS_START = 15
_DEF_SAMPLER = 'TPESampler'
_DEF_WHICH_EMBED = 'CNN'

_EARLY_STOP = 60

# Currently hardcoded, I'll figure out a way in the future
#_DEF_WHICH = ['Om0', 'sigma8', 'm_nu', 'H0', 'ns', 'w0', 'wa']
#bounds = {
#    "Om0":    (0.15, 0.45),   # matter density today
#    "sigma8": (0.60, 1.00),
#    "m_nu":   (0.00, 0.60),   # eV, sum of neutrino masses (adjust if you want a nonzero minimum)
#    "H0":     (55.0, 80.0),   # km/s/Mpc
#    "ns":     (0.90, 1.05),
#    "w0":     (-1.30, -0.70),
#    "wa":     (-0.90, 0.90),
#}
_DEF_WHICH = ['Om0', 'sigma8', 'h']
# Std Sobol sequence
bounds = {
    "Om0":    (0.1, 0.5),
    "sigma8": (0.60, 1.20),
    "h":      (0.60, 0.80),
}
# Dense Sobol sequence
#bounds = {
#    "Om0":    (0.29, 0.33),
#    "sigma8": (0.77, 0.79),
#    "h":      (0.68, 0.72),
#}

# sbi moves the data from cpu->cuda independently, I simply suppress this unnecessary warning
warnings.filterwarnings("ignore", message=".*Moving.*to the data_device.*", category=UserWarning)

def printlog(text: str, flush: bool = False):
    now = datetime.now().strftime('%H:%M:%S')
    logger = 'TUNE'
    print(f'\033[92m[{logger:<5}: {now}]>\033[0m ' + text, flush=flush)

def printsbi(text: str, flush: bool = False):
    now = datetime.now().strftime('%H:%M:%S')
    logger = 'SBI'
    print(f'\033[92m[{logger:<5}: {now}]>\033[0m ' + text, flush=flush)

# ======================================================================================== #
# Trial
# ======================================================================================== #

def _suggest_categorical(trial, cat_arguments, cat_name):
    """
    Suggest a categorical value for the trial if cat_arguments contains more than one value

    (helper function)
    """
    if len(cat_arguments) > 1:
        value = trial.suggest_categorical(cat_name, cat_arguments)
    else:
        value = cat_arguments[0]

    return value

def suggest_values(trial, args):
    """
    Suggests values to test in the trial and returns a dictionary containing these values
    """
    # Hyperparameters to tune
    values_dict = {}

    values_dict['dens_est'] = _suggest_categorical(trial, args.density_est, 'density_estimator')
    values_dict['batchsize'] = _suggest_categorical(trial, args.batchsize, 'batchsize')

    if args.use_embedding:

        if args.which_embedding == 'CNN':
            values_dict['cnn_hidden_features'] = _suggest_categorical(trial, args.cnn_hidden_features, 'cnn_hidden_features')
        if args.which_embedding == 'FC':
            values_dict['fc_layers'] = trial.suggest_int('fc_layers', 1, args.fc_layers_max)
            values_dict['fc_halve'] = trial.suggest_categorical('fc_halve', [True, False])
        
        values_dict['fc_hidden_features'] = _suggest_categorical(trial, args.fc_hidden_features, 'fc_hidden_features')
        values_dict['fc_out_dimension'] = _suggest_categorical(trial, args.fc_out_dimension, 'fc_out_dimension')

    values_dict['num_transforms'] = trial.suggest_int('num_transforms', 2, args.num_transforms_max)
    values_dict['hidden_features'] = trial.suggest_categorical('hidden_features', args.hidden_features)
    values_dict['learning_rate'] = trial.suggest_float('learning_rate', args.learning_rate_min, args.learning_rate_max, log=True)
    values_dict['normalise_labels'] = trial.suggest_categorical('normalise_labels', [True, False])

    printlog(f'Starting configuration (trial {trial.number})', flush=True)
    msg = '; '.join(f'{key}={val}' for key, val in values_dict.items())
    printlog(msg, flush=True)

    return values_dict

# ======================================================================================== #
# Objective function
# ======================================================================================== #

def objective(trial, args):

    suggested_values = suggest_values(trial, args)
    
    # device
    if torch.cuda.is_available():
        device = torch.device('cuda',0)
    else:
        device = 'cpu'
    printsbi(f'Running on device {device}', flush=True)

    # define prior
    which_idx, lowb, highb = select_pars(bounds, args.which_param, normalise=suggested_values['normalise_labels'])
    prior = BoxUniform(low = lowb, high = highb, device=device)

    if args.k_crossval == 0:
        # do not perform k cross validation

        # define dataset
        dat, SHAPE = load_dataset(args.parameters_path, args.sumstats_path, args.dataset_type, verb=args.first_trial,
                                  bounds=bounds, normalise=suggested_values['normalise_labels'], return_shape=True,
                                  flatten=args.flatten
                                  )
        if args.do_pca:
            dat.pca(ncomp=3, verb=args.first_trial)
        train, val, _ = dat.split3(args.val_frac, args.test_frac, verb=args.first_trial)
        args.first_trial = False

        N_POSTERIORS = args.n_ensemble if args.ensemble_training else 1
        posteriors = []

        for _ in range(N_POSTERIORS):

            if args.use_embedding and args.which_embedding == 'CNN':
                embedding_net = MatrixEmbedding(
                        in_shape = SHAPE,
                        output_dim = suggested_values['fc_out_dimension'],
                        cnn_init_hidden_chn = suggested_values['cnn_hidden_features'],
                        fc_hidden_chn = suggested_values['fc_hidden_features'],
                        )
            elif args.use_embedding and args.which_embedding == 'FC':
                embedding_net = FCEmbedding(
                        in_channels = SHAPE[0],
                        out_channels = suggested_values['fc_out_dimension'],
                        hidden_channels = suggested_values['fc_hidden_features'],
                        n_layers = suggested_values['fc_layers'],
                        halve = suggested_values['fc_halve'],
                        )
            else:
                embedding_net = nn.Identity()

            density_estimator = posterior_nn(
                    model = suggested_values['dens_est'],
                    num_transforms = suggested_values['num_transforms'],
                    hidden_features = suggested_values['hidden_features'],
                    embedding_net = embedding_net,
                    )

            inference = NPE(prior = prior, density_estimator = density_estimator, device = device,
                            tracker = None, show_progress_bars = False)
            inference.append_simulations(train.theta[:,which_idx], train.x_pca if args.do_pca else train.x)
            
            # catch the stdout
            captured = io.StringIO()
            init_time = datetime.now()
            with contextlib.redirect_stdout(captured):
                estimator = inference.train(
                        training_batch_size = suggested_values['batchsize'],
                        learning_rate = suggested_values['learning_rate'],
                        show_train_summary = False,
                        )
            sbi_output = captured.getvalue().strip()
            if sbi_output:  printsbi(str(sbi_output), flush=True)
            printsbi(f'Elapsed time: {datetime.now() - init_time}', flush=True)

            posteriors.append(inference.build_posterior(estimator))

        # works for one or more
        ensemble_posterior = EnsemblePosterior(posteriors)

        with torch.no_grad():
            del train
            val.to(device)
            # norm_posterior to False for faster evaluations which are fine for HPO
            return_value = -ensemble_posterior.log_prob_batched(val.theta[:,which_idx], x = val.x_pca if args.do_pca else val.x, norm_posterior=False).mean().item()

        del ensemble_posterior, posteriors, val, inference, density_estimator

    else: # args.k_crossval > 0

        nlps = []
        
        # define dataset
        dat, SHAPE = load_dataset(args.parameters_path, args.sumstats_path, args.dataset_type, verb=args.first_trial,
                                  bounds=bounds, normalise=suggested_values['normalise_labels'], return_shape=True,
                                  flatten=args.flatten
                                  )
        if args.do_pca:
            dat.pca(ncomp=3, verb=args.first_trial)
        train, _ = dat.split(args.test_frac, verb=args.first_trial)
        args.first_trial = False

        N = len(train)              # total number of sims for cross-validation
        Nk = N // args.k_crossval   # number of elements per cross-validation step

        for i in range(args.k_crossval):

            printsbi(f'Doing k-fold cross-validation {i+1}/{args.k_crossval} (trial {trial.number})', flush=True)
            
            # running mask for the validation set
            val_start = i * Nk
            val_end = val_start + Nk if i < args.k_crossval - 1 else N

            val_mask = torch.zeros(N, dtype=torch.bool)
            val_mask[val_start:val_end] = True
            train_mask = ~val_mask

            train_theta = train.theta[train_mask][:, which_idx]
            train_x = train.x_pca[train_mask] if args.do_pca else train.x[train_mask]
            val_theta = train.theta[val_mask][:, which_idx]
            val_x = train.x_pca[val_mask] if args.do_pca else train.x[val_mask]

            if args.use_embedding and args.which_embedding == 'CNN':
                embedding_net = MatrixEmbedding(
                        in_shape = SHAPE,
                        output_dim = suggested_values['fc_out_dimension'],
                        cnn_init_hidden_chn = suggested_values['cnn_hidden_features'],
                        fc_hidden_chn = suggested_values['fc_hidden_features'],
                        )
            elif args.use_embedding and args.which_embedding == 'FC':
                embedding_net = FCEmbedding(
                        in_channels = SHAPE[0],
                        out_channels = suggested_values['fc_out_dimension'],
                        hidden_channels = suggested_values['fc_hidden_features'],
                        n_layers = suggested_values['fc_layers'],
                        halve = suggested_values['fc_halve'],
                        )
            else:
                embedding_net = nn.Identity()

            density_estimator = posterior_nn(
                    model = suggested_values['dens_est'],
                    num_transforms = suggested_values['num_transforms'],
                    hidden_features = suggested_values['hidden_features'],
                    embedding_net = embedding_net,
                    )

            inference = NPE(prior = prior, density_estimator = density_estimator, device=device,
                            tracker = None, show_progress_bars = False)
            inference.append_simulations(train_theta, train_x)
            
            # catch the stdout
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                estimator = inference.train(
                        training_batch_size = suggested_values['batchsize'],
                        learning_rate = suggested_values['learning_rate'],
                        show_train_summary = False,
                        stop_after_epochs = _EARLY_STOP,
                        )
            sbi_output = captured.getvalue().strip()

            posterior = inference.build_posterior(estimator)

            with torch.no_grad():
                nlp = -posterior.log_prob_batched(val_theta.to(device), x = val_x.to(device), norm_posterior=False).mean().item()

            nlps.append(nlp)
            printsbi(f'{i+1} finished with {nlp:.4e}', flush=True)

        return_value = np.mean(nlps)
        printsbi(f'Mean neglog-posterior for trial {trial.number}: {return_value:.4e}', flush=True)

    return return_value


if __name__ == '__main__':

    # ======================================================================================== #
    # Args
    # ======================================================================================== #

    parser = argparse.ArgumentParser(description='Compute different summary statistics for the post-processed Pinocchio dataset')

    parser.add_argument('--directory', type=str, required=True)
    parser.add_argument('--study-name', type=str, required=True)
    parser.add_argument('--dataset-type', type=str, required=True)
    parser.add_argument('--parameters-path', type=str, required=True)
    parser.add_argument('--sumstats-path', type=str, required=True)

    parser.add_argument('--val-frac', type=float, default=_DEF_VALFR)
    parser.add_argument('--test-frac', type=float, default=_DEF_TESTFR)
    parser.add_argument('--which-param', type=str, nargs='+', default=_DEF_WHICH)
    parser.add_argument('--do-pca', action='store_true')
    parser.add_argument('--esemble-training', action='store_true')
    parser.add_argument('--n-ensemble', type=int, default=_DEF_N_ENSEMBLE)
    parser.add_argument('--k-crossval', type=int, default=0)

    parser.add_argument('--density-est', type=str, nargs='+', default=_DEF_DENS_EST)
    parser.add_argument('--num-transforms-max', type=int, default=_DEF_NUM_TRANSF_MAX)
    parser.add_argument('--hidden-features', type=int, nargs='+', default=_DEF_HIDDENFEAT)
    parser.add_argument('--batchsize', type=int, nargs='+', default=_DEF_BATCHSIZE)
    parser.add_argument('--learning-rate-min', type=float, default=_DEF_LR_MIN)
    parser.add_argument('--learning-rate-max', type=float, default=_DEF_LR_MAX)

    parser.add_argument('--use-embedding', action='store_true')
    parser.add_argument('--which-embedding', type=str, default=_DEF_WHICH_EMBED)
    parser.add_argument('--cnn-hidden-features', type=int, nargs='+', default=_DEF_CNN_HIDDENFEAT)
    parser.add_argument('--fc-hidden-features', type=int, nargs='+', default=_DEF_FC_HIDDENFEAT)
    parser.add_argument('--fc-out-dimension', type=int, nargs='+', default=_DEF_FC_OUTDIM)
    parser.add_argument('--fc-layers-max', type=int, default=_DEF_FC_LAYERS_MAX)

    parser.add_argument('--sampler', type=str, default=_DEF_SAMPLER)
    parser.add_argument('--n-trials', type=int, default=_DEF_NTRIALS)
    parser.add_argument('--n-startup-trials', type=int, default=_DEF_NTRIALS_START)

    args = parser.parse_args()

    if args.do_pca and args.use_embedding:
        raise AttributeError('Cannot use both PCA and an embedding network')
    if args.ensemble_training and args.k_crossval > 0:
        raise AttributeError('Cannot use both ensemble training and k-cross validation currently')

    if args.which_embedding == 'FC':
        args.flatten = True
    else:
        args.flatten = False

    # ======================================================================================== #

    printlog('NPE tuning')
    printlog(f'Found boundaries: {str(bounds)}')
    printlog('== Parsed values ==')
    for arg in vars(args):
        printlog(f'{arg:<20} {getattr(args, arg)}')

    dirname = f'{args.directory}/optuna/'
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
        printlog(f'Created directory {dirname}')

    args.first_trial = True # to print some informations only at the first trial
    
    # ======================================================================================== #

    storage = f"sqlite:///{dirname}/{args.study_name}.db"

    study = optuna.create_study(study_name=args.study_name,
                                storage=storage,
                                load_if_exists=True,
                                direction='minimize',
                                sampler=getattr(optuna.samplers, args.sampler)(n_startup_trials=args.n_startup_trials),
                                )

    study.optimize(partial(objective, args=args), n_trials=args.n_trials, catch=(torch.OutOfMemoryError))

    # Print the results
    printlog(f"Best hyperparameters: {str(study.best_trial.params)}")
