"""
PINSBI: diagnostics

This script provides useful functions to help diagnose the inference
"""
import numpy as np
import matplotlib.pyplot as plt
import torch

from captum.attr import Saliency, IntegratedGradients
from joblib import Parallel, delayed
from scipy.stats import gaussian_kde
from tqdm import tqdm

def posterior_zscore(npe_samples, lbi_samples, param_names=None, credible_levels=(0.68, 0.95)):
    """
    Compute z-scores between NPE and LBI posteriors for each parameter.
    Also computes the ratio of posterior widths (NPE/LBI) as a direct
    check of which is broader.

    Args:
        npe_samples (np.ndarray): array containing the SBI samples
        lbi_samples (np.ndarray): array containing the LBI samples
        param_names (list[str], default=None): list of parameters names
        credible_levels (tuple, default=(0.68,0.95)): credible levels to check

    Returns:
        results (dict)
    """
    n_params = npe_samples.shape[1]
    if param_names is None:
        param_names = [f'θ_{i}' for i in range(n_params)]

    results = {}
    for k, name in enumerate(param_names):
        s_npe = npe_samples[:, k]
        s_lbi = lbi_samples[:, k]

        mu_npe, sig_npe = s_npe.mean(), s_npe.std()
        mu_lbi, sig_lbi = s_lbi.mean(), s_lbi.std()

        # z-score on means
        z = (mu_npe - mu_lbi) / np.sqrt(sig_npe**2 + sig_lbi**2)

        # width ratio: > 1 means NPE is broader
        width_ratio = sig_npe / sig_lbi

        # credible interval comparison
        ci = {}
        for cl in credible_levels:
            tail = (1 - cl) / 2
            ci_npe = np.quantile(s_npe, [tail, 1 - tail])
            ci_lbi = np.quantile(s_lbi, [tail, 1 - tail])
            ci[cl] = {
                'npe': ci_npe,
                'lbi': ci_lbi,
                'width_npe': ci_npe[1] - ci_npe[0],
                'width_lbi': ci_lbi[1] - ci_lbi[0],
                'width_ratio': (ci_npe[1] - ci_npe[0]) / (ci_lbi[1] - ci_lbi[0]),
            }

        results[name] = {
            'mu_npe': mu_npe, 'sig_npe': sig_npe,
            'mu_lbi': mu_lbi, 'sig_lbi': sig_lbi,
            'z': z,
            'width_ratio': width_ratio,
            'ci': ci,
        }

        print(f"\n{name}:")
        print(f"  NPE:  μ = {mu_npe:.4f},  σ = {sig_npe:.4f}")
        print(f"  LBI:  μ = {mu_lbi:.4f},  σ = {sig_lbi:.4f}")
        print(f"  z-score: {z:.3f}  ({'consistent' if abs(z)<2 else 'TENSION'})")
        print(f"  σ_NPE / σ_LBI = {width_ratio:.3f}  "
              f"({'NPE broader' if width_ratio>1 else 'LBI broader'})")
        for cl, vals in ci.items():
            print(f"  {int(cl*100)}% CI width ratio: {vals['width_ratio']:.3f}")

    return results

def compute_coverage_generic(testdata, posterior_sampler, which_idx, num_samples: int = 10000,
                             n_alpha: int = 10, num_workers: int = -1, use_mahal: bool = False,
                             sel_idx = None):
    """

    Args:
        testdata (SBIDataset)
        posterior_sampler (callable): sampler such that idx -> (num_samples, D) posterior samples
        which_idx (list[int]): which parameters to sample
        num_samples (int, default=10000): posterior samples per observation
        n_alpha (int, default=10): number of credibility levels
        num_workers (int, default=-1): number of parallel workers
        use_mahal (bool, default=False): use the Mahalonobis distance instead of gaussian KDE to compute
            the joint HPD coverage
        sel_idx (array-like of int or bool mask, default=None): subset of test indices to use.
            If None, all test elements are used

    Returns:
        alphas            : (n_alpha,) credibility levels
        marginal_coverage : (n_alpha, D) empirical coverage per parameter
        joint_coverage    : (n_alpha,) joint HPD coverage
    """
    def process_one(idx, testdata, which_idx, num_samples, alphas, posterior_sampler):
        x_obs      = testdata.x[idx]
        theta_true = testdata.theta[idx, which_idx]       # (D,)
        posterior  = posterior_sampler(num_samples, x_obs) # (num_samples, D)
        D          = theta_true.shape[0]

        # marginal coverage (per parameter)
        marginal = np.zeros(( len(alphas), D))
        for d in range(D):
            rank = np.searchsorted(np.sort(posterior[:, d]), theta_true[d])
            frac = rank / num_samples
            marginal[:, d] = (alphas >= frac)

        if use_mahal:
            # joint coverage via Mahalonobis distance
            try:
                post = posterior.cpu().numpy().astype(np.float64)
                tt = theta_true.cpu().numpy().astype(np.float64)
                mu  = post.mean(axis=0)
                cov = np.cov(post.T)
                cov_inv = np.linalg.inv(cov + 1e-8 * np.eye(len(mu)))

                def mahal(x):
                    diff = x - mu
                    return diff @ cov_inv @ diff

                dists = np.array([mahal(s) for s in post])
                dist_true = mahal(tt)
                hpd_level = np.mean(dists >= dist_true)
                joint = (alphas >= 1 - hpd_level).astype(float)

            except Exception:
                joint = np.full(len(alphas), np.nan)

        else:
            # joint coverage via HPD
            try:
                kde = gaussian_kde(posterior.T)
                log_density      = kde.logpdf(posterior.T)
                log_density_true = kde.logpdf(theta_true[:, None])[0]
                hpd_level        = np.mean(log_density <= log_density_true)
                joint            = (alphas >= 1 - hpd_level).astype(float)
            except Exception:
                joint = np.full(len(alphas), np.nan)

        return marginal, joint

    alphas  = np.linspace(0, 1, n_alpha)
    n_total = testdata.theta.shape[0]
    D       = testdata.theta[:, which_idx].shape[1]

    # resolve the subselection
    if sel_idx is None:
        indices = np.arange(n_total)
    else:
        sel_idx = np.asarray(sel_idx)
        if sel_idx.dtype == bool:
            if sel_idx.shape[0] != n_total:
                raise ValueError(f'Boolean sel_idx has length {sel_idx.shape[0]}; expected {n_total}')
            indices = np.flatnonzero(sel_idx)
        else:
            indices = sel_idx.astype(int).ravel()
        if indices.size == 0:
            raise ValueError('sel_idx selects no elements')
        if indices.min() < 0 or indices.max() >= n_total:
            raise IndexError(f'sel_idx contains indices outside [0, {n_total - 1}]')

    n_test = len(indices)

    results = Parallel(n_jobs=num_workers)(
            delayed(process_one)(idx, testdata, which_idx, num_samples, alphas, posterior_sampler)
            for idx in tqdm(indices)
            )

    marginal_counts = np.zeros((n_alpha, D))
    joint_counts    = np.zeros(n_alpha)
    for marginal, joint in results:
        marginal_counts += marginal
        joint_counts    += joint

    marginal_coverage = marginal_counts / n_test
    joint_coverage    = joint_counts    / n_test
    return alphas, marginal_coverage, joint_coverage

def compute_covariance_posterior(testdata, posterior, which_use: list[str], bounds: dict, n_samples: int = 10000, normalise: bool = False, num_workers: int = -1):
    """
    Compute a mean covariance matrix of the given posterior parameters across the passed data set
    """
    from pinsbi.utils import sample_npe

    N = testdata.x.shape[0]
    D = len(which_use)
    model_numbers = testdata.index

    cov = np.zeros((N,D,D))
    corr = np.zeros((N,D,D))

    def process_one(idx, run_number):
            samples, _ = sample_npe(testdata, posterior, run_number, which_use, bounds=bounds, n_samples=n_samples, normalise=normalise, verb=False) # (n_samples, D)
            return np.cov(samples.T), np.corrcoef(samples.T)


    results = Parallel(n_jobs=num_workers)(
            delayed(process_one)(idx, run_number)
            for idx, run_number in tqdm(enumerate(model_numbers))
            )

    cov_all, corr_all = zip(*results)
    cov = np.array(cov_all)
    corr = np.array(corr_all)

    m_cov = np.mean(cov, axis=0)
    s_cov = np.std(cov, axis=0)
    m_corr = np.mean(corr, axis=0)
    s_corr = np.std(corr, axis=0)

    return m_cov, s_cov, m_corr, s_corr


def compute_saliency_maps(embedding_net, density_estimator, test_dataset, run_number,
                           method="ig", n_steps=50, baseline=None):
    """
    Compute saliency maps for a trained CNN embedding net + NPE density estimator.

    Args:
        embedding_net (nn.Module): Trained CNN embedding net, same object used
            inside `density_estimator`.
        density_estimator (nn.Module): Trained flow. Must expose `.log_prob(theta, x)`.
        test_dataset (SBIDataset): test dataset.
        run_number (int): run number to test.
        method (string, {"saliency", "ig"}): Attribution method: plain gradients ("saliency")
            or IntegratedGradients ("ig", recommended -- less noisy, satisfies completeness).
        n_steps (int, default=50): Number of integration steps for IntegratedGradients (ignored for plain saliency).
        baseline (torch.Tensor or None): Baseline input for IntegratedGradients. Defaults to an all-zeros
            input of the same shape as x0.

    Returns:
        embedding_map (np.ndarray, shape (H, W)): Mean absolute attribution across all embedding output dimensions --
            which input pixels drive the learned representation overall.
        embedding_maps_per_dim (np.ndarray, shape (n_embed_dims, H, W)): Per-dimension attribution, in case you want to 
            inspect individual latent features rather than the aggregate.
        posterior_map (np.ndarray, shape (H, W)): Attribution of log p(theta_true | x) w.r.t. the input -- which
            pixels matter for inference of the true parameters specifically.
    """
    embedding_net.eval()
    density_estimator.eval()
    for p in embedding_net.parameters():
        p.requires_grad_(False)
    for p in density_estimator.parameters():
        p.requires_grad_(False)

    run_idx = np.where(test_dataset.index == run_number)[0]
    if run_idx.size == 0:
        raise IndexError(f'run number {run_number} not found in the dataset')
    run_idx = run_idx[0]
    x0 = test_dataset.x[run_idx].clone().float().unsqueeze(0)
    x0.requires_grad_(True)

    theta_true = torch.as_tensor(test_dataset.theta[run_idx], dtype=torch.float32)

    if baseline is None:
        baseline = torch.zeros_like(x0)

    # embedding dimensions
    n_embed_dims = embedding_net(x0).shape[-1]
    embedding_maps_per_dim = np.zeros((n_embed_dims, *x0.shape[-2:]))

    if method == "saliency":
        attributor = Saliency(embedding_net)
    elif method == "ig":
        attributor = IntegratedGradients(embedding_net)
    else:
        raise ValueError(f"Unknown method: {method}")

    for dim in range(n_embed_dims):
        if method == "saliency":
            attr = attributor.attribute(x0, target=dim)
        else:
            attr = attributor.attribute(x0, target=dim, baselines=baseline, n_steps=n_steps)
        embedding_maps_per_dim[dim] = attr.squeeze().detach().cpu().numpy()

    embedding_map = np.mean(np.abs(embedding_maps_per_dim), axis=0)

    # posterior log-density
    def log_prob_fn(x_batch):
        lp = density_estimator.log_prob(theta_true.expand(x_batch.shape[0], -1), x_batch)
        return lp.squeeze(0)  # (sample_dim=1, batch_dim) -> (batch_dim,)

    if method == "saliency":
        post_attributor = Saliency(forward_func=log_prob_fn)
        attr = post_attributor.attribute(x0)
    else:
        post_attributor = IntegratedGradients(forward_func=log_prob_fn)
        attr = post_attributor.attribute(x0, baselines=baseline, n_steps=n_steps)

    posterior_map = attr.squeeze().detach().cpu().numpy()

    return embedding_map, embedding_maps_per_dim, posterior_map
