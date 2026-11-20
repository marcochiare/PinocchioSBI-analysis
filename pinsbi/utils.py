"""
PINSBI: utils

This script provides commonly used functions for the analysis
"""
import glob
import numpy as np
import os
import torch
import warnings

from pinsbi.dataset import SBIDataset, EmuDataset
from tqdm import tqdm

from sbi.diagnostics import run_tarp

def normalize_param(params: np.ndarray, param_name: str, bounds: dict):
    """
    Normalises the cosmological parameters

    Args:
        params (np.ndarray, shape=(N,)): values array corresponding to one cosmological parameter
        param_name (str): corresponding cosmological parameter to normalise
        bounds (dict[str:tuple]): dictionary mapping a cosmological parameter to a tuple containing
            its low and high boundaries

    Returns:
        params (np.ndarray, shape=(N,)): normalised array
    """
    assert param_name in bounds.keys(), f'{param_name} is not present in `bounds` keys.'

    norm_params = (params - bounds[param_name][0])/(bounds[param_name][1] - bounds[param_name][0])

    return norm_params

def denormalize_param(params: np.ndarray, param_name: str, bounds: dict):
    """
    Restores the normalisation of cosmological parameters

    Args:
        params (np.ndarray, shape=(N,)): values array corresponding to one cosmological parameter
        param_name (str): corresponding cosmological parameter to de-normalise
        bounds (dict[str:tuple]): dictionary mapping a cosmological parameter to a tuple containing
            its low and high boundaries

    Returns:
        params (np.ndarray, shape=(N,)): array with restored normalisation
    """
    assert param_name in bounds.keys(), f'{param_name} is not present in `bounds` keys.'

    restored_params = (bounds[param_name][1] - bounds[param_name][0]) * params + bounds[param_name][0]
   
    return restored_params

def tarp_bootstrap(theta, x, posterior, n_bootstraps: int = 50, num_samples: int = 1000, bins: int = None, desc: str =  None, **tarp_kwargs):
    """
    Run multiple computations of TARP to produce error bars

    Args:
        theta, x: pair of parameter-observation
        posterior: the trained sbi posterior
        n_bootstraps (int, default=50): number of computations
        num_samples (int): how many samples to generate for each computation
        bins (int, default=None): number of bins for the expected coverage probability
        desc (str, default=None): progressive bar description
        **tarp_kwargs: extra kwargs to pass to run_tarp

    Returns:
        ecps (np.ndarray, shape=(nsamples, nbins)
        alpha
    """
    ecps = []

    for i in tqdm(range(n_bootstraps), desc=desc):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            ecp, alpha = run_tarp(
                theta,
                x,
                posterior,
                references=None,
                num_posterior_samples=num_samples,
                use_batched_sampling=False,  # `True` can give speed-ups, but can cause memory issues.
                show_progress_bar=False,
                num_bins=bins,
                **tarp_kwargs
            )

            ecps.append(ecp)

    return np.array(ecps), alpha

def select_pars(bounds, which, normalise: bool = False):
    """
    Given a "bounds" dictionary containing pairs of "cosmo_par": (lowbound, highbound), and a list "which" of strings
    containing the requested cosmo parameters, returns the indexing mask and the corresponding lowbounds and highbounds
    torch tensors.

    If normalise=True, the low and high boundaries are between zeros and one.
    """
    param_order = list(bounds.keys())
    param_to_idx = {p: i for i, p in enumerate(param_order)}
    which_idx = [param_to_idx[p] for p in which]

    lowbound = torch.tensor([ bounds[key][0] for key in which ], dtype=torch.float32)
    highbound = torch.tensor([ bounds[key][1] for key in which ], dtype=torch.float32)

    if normalise:
        lowbound = torch.zeros_like(lowbound)
        highbound = torch.ones_like(highbound)

    return which_idx, lowbound, highbound

def load_dataset(param_path: str, stats_path: str, probe: str, normalise: bool = False, bounds: dict[tuple] = None, type_norm: str = 'log',
                 return_norm: bool = False, return_shape: bool = False, flatten: bool = False, save_prefix: str = 'model', verb: bool = True):
    """
    Create a SBIDataset object that combines parameters and summary statistics from the provided paths.
    Returns a ready-for-training version of the dataset, with normalisation/log applied

    Args:
        param_path (str): path to the true parameters file.
        stats_path (str): path to the directory containing the summary statistics.
        probe (str): type of probe to read, as saved in the summary statistics.
        normalise (bool, default=False): normalise the labels between 0 and 1.
        bounds (dict[tuple], default=None): low/high boundaries for each parameter. 
            Necessary if the parameters are being normalised.
        type_norm (str, default='log'): 'log' for log10(1+N) or 'ans' for Anscombe.
        return_norm (bool, default=False): return the normalisation of the dataset.
        return_shape (bool, default=False): return the shape of the dataset.
        flatten (bool, default=False): flatten the summary statistics matrix.
        save_prefix (str, default='model'): base name of the files to read.
        verb (bool, default=True): verbosity.

    Returns:
        SBIDataset: dataset class.
        norm: normalisation factor, if return_norm is True.
        shape: shape of the number counts array, if return_shape is True.
    """
    if type_norm not in ['log', 'ans']:
        raise ValueError(f'Provided type of normalisation {type_norm!r} is not valid. Use `log` or `ans`.')

    if normalise and bounds is None:
        raise ValueError('Specify a dictionary that provides low/high boundaries for normalisation')

    if not os.path.isfile(param_path):
        raise FileNotFoundError(f'File {param_path} not found.')
    if not os.path.isdir(stats_path):
        raise FileNotFoundError(f'Directory {stats_path} not found.')

    # Getting the files name and sorting them correctly
    get_idx = lambda f: os.path.basename(f)[len(save_prefix)+1 : -len('.npz')]
    allfiles = glob.glob(f'{stats_path}/{save_prefix}_*.npz')
    files = sorted([f for f in allfiles if get_idx(f).isdigit()], key=lambda f: int(get_idx(f)))
    idx = np.array([int(get_idx(f)) for f in files])
    skipped = sorted(set(allfiles) - set(files))
    Ntotal = len(files) 
    assert Ntotal > 0, f'No `{save_prefix}_<int>.npz` files found in {stats_path}'

    allpars = np.loadtxt(param_path, ndmin=2)
    missing = np.setdiff1d(np.arange(idx.max() + 1), idx)
    assert idx.max() < len(allpars), 'file index beyond the parameter file'
    assert len(np.unique(idx)) == len(idx), 'duplicate file indices'
    # each file is paired with its own row of the parameters file
    pars = allpars[idx].copy()
    if normalise:
        assert len(bounds) <= pars.shape[1], f'Found {len(bounds)} boundaries but only {pars.shape[1]} parameters in {param_path}'
        for i, key in enumerate(bounds.keys()):
            pars[:,i] = normalize_param(pars[:,i], key, bounds)
        outside = [key for i, key in enumerate(bounds.keys()) if np.any((pars[:,i] < 0) | (pars[:,i] > 1))]
 
    if verb:
        print(f'Loading dataset for probe "{probe}"')
        print(f'true parameters path:  {param_path}')
        print(f'summ. statistics path: {stats_path}')
        print(f'files found:           {Ntotal}')
        if normalise:   print('True labels are normalised between (0, 1)')
    if len(skipped):    print(f'WARNING: {len(skipped)} skipped files, e.g. {skipped[:3]}')
    if len(missing):    print(f'WARNING: {len(missing)} missing files, e.g. {missing[:10]}')
    if normalise and outside:   print(f'WARNING: true labels outside the boundaries for {outside}')
 
    # reading the files
    dataset = []
    try:
        for file in files:
            with np.load(file) as f:
                dataset.append(f[probe])
    except Exception as e:
        raise ValueError(f"{e} in file: {file}")
 
    # all the summary statistics must share the same shape
    shapes = {d.shape for d in dataset}
    if len(shapes) > 1:
        raise ValueError(f'Found summary statistics with different shapes: {shapes}')
 
    dataset = np.array(dataset, dtype=np.float64)
 
    # dataset checks
    if not np.all(np.isfinite(dataset)):
        raise ValueError('Found non-finite values in the summary statistics.')
    if probe in ['counts_z','counts_mass_z','counts_lmb_z'] and dataset.min() < 0:
        raise ValueError(f'Found negative number counts (min {dataset.min():.3g}).')
 
    # log count
    if probe in ['counts_z']:
        dataset = np.log10(dataset)
        if verb: print('Applied log10(counts)')
    elif probe in ['counts_mass_z','counts_lmb_z'] and type_norm == 'log':
        dataset = np.log10(1+dataset) # prevent -inf
        if verb: print('Applied log10(1+counts)')
    elif probe in ['counts_mass_z','counts_lmb_z'] and type_norm == 'ans':
        dataset = 2 * np.sqrt(dataset + 3./8) # anscombe transformation
        if verb: print('Applied 2*sqrt(counts + 3/8)')
    else:
        if verb: print(f'No log applied to {probe}')
 
    # flatten into a vector if a matrix (ndim=3)
    if dataset.ndim == 3 and flatten:
        if verb: print('Flattening the input vector')
        dataset = dataset.reshape(dataset.shape[0], -1)
    # to return the matrix or vector shape
    SHAPE = dataset.shape[1:]
 
    # normalization
    NORM = dataset.max()
    dataset /= NORM
    if verb:    print('Features normalisation value:', NORM)
 
    # dataset first, then the normalisation and/or the shape (if requested)
    out = (SBIDataset(idx, pars, dataset),)
    if return_norm:     out += (NORM,)
    if return_shape:    out += (SHAPE,)
 
    return out if len(out) > 1 else out[0]

def load_dataset_repl(stats_path: str, probe: str, normalise: bool = False, bounds: dict[tuple] = None, type_norm: str = 'log',
                      return_norm: bool = False, return_shape: bool = False, verb: bool = True):
    """
    Create a SBIDataset object that combines parameters and summary statistics from the provided paths
    considering multiple realisation for each cosmological model.
    Returns a ready-for-training version of the dataset, with normalisation/log applied

    NB: each summary statistics file must contain the parameters as dictionary entries.
        Be sure that the `bounds` dict keys coincide with these ones.

    Args:
        stats_path (str): path to the directory containing the summary statistics.
        probe (str): type of probe to read, as saved in the summary statistics.
        normalise (bool, default=False): normalise the labels between 0 and 1.
        bounds (dict[tuple], default=None): low/high boundaries for each parameter. 
            Necessary if the parameters are being normalised.
        type_norm (str, default='log'): 'log' for log10(1+N) or 'ans' for Anscombe.
        return_norm (bool, default=False): return the normalisation of the dataset.
        return_shape (bool, default=False): return the shape of the dataset before flattening
        verb (bool, default=True): verbosity.

    Returns:
        SBIDataset: dataset class.
        norm: normalisation factor, if return_norm is True.
    """
    assert type_norm in ['log', 'ans'], f'Provided type of normalisation {type_norm} is not valid. Use `log` or `ans`.'

    if normalise:
        assert bounds is not None, 'Specify a dictionary that provides low/high boundaries for normalisation'

    if not os.path.isdir(stats_path):
        raise FileNotFoundError(f'Directory {stats_path} not found.')

    # Getting the files name and sorting them correctly
    files = sorted(glob.glob(f'{stats_path}/model_*.npz'), key=lambda f: int(f.split('model_')[-1].split('.npz')[0]))
    Ntotal = len(files)

    if verb:
        print(f'Loading dataset for probe "{probe}"')
        print(f'summ. statistics path: {stats_path}')
        if normalise:   print('True labels are normalised between (0, 1)')

    dataset = []
    pars = []
    for file in files:

        cat_file = np.load(file)
        dataset.append(cat_file[probe])

        cat_pars = []
        for key in bounds.keys():
            cat_pars.append(cat_file[key])

        pars.append(cat_pars)

    dataset = np.array(dataset)
    pars = np.array(pars)

    # normalise between (0,1) the true cosmological parameters
    if normalise:
        for i, key in enumerate(bounds.keys()):
            pars[:,i] = normalize_param(pars[:,i], key, bounds)

    # log count
    if probe in ['counts_z']:
        dataset = np.log10(dataset)
        if verb: print('Applied log10(counts)')
    elif probe in ['counts_mass_z','counts_lmb_z'] and type_norm == 'log':
        dataset = np.log10(1+dataset) # prevent -inf
        if verb: print('Applied log10(1+counts)')
    elif probe in ['counts_mass_z','counts_lmb_z'] and type_norm == 'ans':
        dataset = 2 * np.sqrt(dataset + 3./8) # anscombe transformation
    else:
        if verb: print(f'No log applied to {probe}')

    # flatten into a vector if a matrix (ndim=3)
    SHAPE = dataset.shape[1:]
    if dataset.ndim == 3:
        dataset = dataset.reshape(dataset.shape[0], -1)

    # normalization
    NORM = dataset.max()
    dataset /= NORM
    if verb:    print('Features normalisation value:', NORM)

    if return_norm:
        return SBIDataset(np.arange(0, Ntotal), pars, dataset), NORM
    elif return_shape:
        return SBIDataset(np.arange(0, Ntotal), pars, dataset), SHAPE
    else:
        return SBIDataset(np.arange(0, Ntotal), pars, dataset)

def load_emudataset(numbercounts_path: str, param_names: list[str], probe: str,
                    normalise_in: bool = False, normalise_out: bool = False, type_norm: str = 'log',
                    bounds: dict[str, tuple] = None, flatten: bool = True, return_info: bool = False,
                    save_prefix: str = 'model', verb: bool = True):
    """
    Create an EmuDataset object: inputs are the cosmological+scaling-relation parameters,
    outputs are the number counts. Parameters are read from each .npz file, so every
    output is paired with its own inputs (missing files only reduce the dataset size).
 
    Args:
        numbercounts_path (str): directory containing the number counts as .npz files.
        param_names (list[str]): (input) parameter names to read from each file.
        probe (str): key of the number counts in the .npz files.
        normalise_in (bool, default=False): normalise the inputs to (0, 1) with `bounds`.
        normalise_out (bool, default=False): transform the number counts with `type_norm`.
        type_norm (str, default='log'): 'log' for log10(1+N) or 'ans' for Anscombe.
        bounds (dict[str, tuple], default=None): (low, high) for each parameter.
            Required if `normalise_in=True`.
        flatten (bool, default=True): flatten each number-counts matrix into a vector.
        return_info (bool, default=False): also return a dict describing the transforms
            (save it with the model weights to reproduce/invert them later).
        save_prefix (str, default='model'): base name of the files to read.
        verb (bool, default=True): verbosity.
 
    Returns:
        EmuDataset, and the info dict if `return_info=True`.
    """
    param_names = list(param_names)
    if normalise_out and type_norm not in ('log', 'ans'):
        raise ValueError(f'Provided type of normalisation {type_norm!r} is not valid. Use `log` or `ans`.')
 
    if normalise_in:
        if bounds is None:
            raise ValueError('Specify `bounds` to normalise the cosmo+scaling-relation parameters.')
        absent = [p for p in param_names if p not in bounds]
        if absent:
            raise KeyError(f'Parameters missing from `bounds`: {absent}. Available: {list(bounds)}')
 
    if not os.path.isdir(numbercounts_path):
        raise FileNotFoundError(f'Directory {numbercounts_path} not found.')
 
    # Getting the files name and sorting them by their integer index.
    get_idx = lambda f: os.path.basename(f)[len(save_prefix)+1 : -len('.npz')]
    allfiles = glob.glob(f'{numbercounts_path}/{save_prefix}_*.npz')
    files = sorted([f for f in allfiles if get_idx(f).isdigit()], key=lambda f: int(get_idx(f)))
    idx = np.array([int(get_idx(f)) for f in files])
    skipped = sorted(set(allfiles) - set(files))
    Ntotal = len(files)

    assert Ntotal > 0, f'No `{save_prefix}_<int>.npz` files found in {numbercounts_path}'
    # NOTE: the parameters are read from each file, so inputs and outputs are always correctly paired.
    # Missing files only reduce the size of the dataset
    missing = np.setdiff1d(np.arange(idx.max() + 1), idx)
    assert len(np.unique(idx)) == len(idx), 'duplicate file indices'
 
    if verb:
        print(f'Loading dataset for probe "{probe}"')
        print(f'number counts path: {numbercounts_path}')
        print(f'input parameters:   {param_names}')
        print(f'files found:        {Ntotal}')
    if len(skipped):    print(f'WARNING: {len(skipped)} skipped files, e.g. {skipped[:3]}')
    if len(missing):    print(f'WARNING: {len(missing)} missing files, e.g. {missing[:10]}')
 
    # reading the files
    dataset = []
    pars = []
    try:
        for file in files:
            with np.load(file) as f:
                dataset.append(f[probe])
                pars.append([f[key].item() for key in param_names])
    except Exception as e:
        raise ValueError(f"{e} in file: {file}")
 
    dataset = np.array(dataset, dtype=np.float64)
    pars = np.array(pars, dtype=np.float64)
    SHAPE = dataset.shape[1:]
 
    if not np.all(np.isfinite(dataset)) or not np.all(np.isfinite(pars)):
        raise ValueError('Found non-finite values in the number counts or in the parameters.')
 
    # In/Out normalisations
    if normalise_in:
        for i, key in enumerate(param_names):
            pars[:,i] = normalize_param(pars[:,i], key, bounds)
        outside = [key for i, key in enumerate(param_names) if np.any((pars[:,i] < 0) | (pars[:,i] > 1))]
        if verb:
            print('Inputs are normalised between (0, 1)')
            if outside:     print(f'WARNING: values outside the boundaries for {outside}')
 
    if normalise_out:
        if dataset.min() < 0:
            raise ValueError(f'Found negative number counts (min {dataset.min():.3g}), cannot apply `{type_norm}`.')
        if type_norm == 'log':
            dataset = np.log10(1+dataset)
            if verb: print('Applied log10(1+counts)')
        elif type_norm == 'ans':
            dataset = 2 * np.sqrt(dataset + 3./8) # anscombe transformation
            if verb: print('Applied 2*sqrt(counts + 3/8)')
 
    # flatten into a vector if a matrix (ndim=3)
    if dataset.ndim == 3 and flatten:
        if verb: print('Flattening the output vector')
        dataset = dataset.reshape(dataset.shape[0], -1)
 
    emudataset = EmuDataset(idx, pars, dataset)
 
    if return_info:
        info = {'probe': probe,
                'param_names': list(param_names),
                'normalise_in': normalise_in,
                'bounds': {key: tuple(bounds[key]) for key in param_names} if normalise_in else None,
                'normalise_out': normalise_out,
                'type_norm': type_norm if normalise_out else None,
                'shape': SHAPE,   # shape of the number counts before flattening
                'flatten': flatten,
                }
        return emudataset, info
    else:
        return emudataset

def sample_npe(test: SBIDataset, posterior, run_number: int, which_use: list[str], bounds: dict, n_samples: int = 10000, normalise: bool = False, verb: bool = True):
    """
    Sample the posterior given an observation

    Args:
        test (SBIDataset): a test dataset with (index, theta, x) methods
        posterior : sbi posterior
        run_number (int): the number of a run to pick (i.e. model_{run_number})
        which_use (list[str]): list of parameters to sample. Names must match the keys in `bounds`
        bounds (dict): a dictionary containing the boundaries for the parameters.
        n_samples (int, default=10000): number of points to sample from the posterior
        normalise (bool, default=False): whether the NPE predicts normalised labels
        verb (bool, default=True): verbosity

    Returns:
        samples (np.ndarray, shape=(n_samples, len(which_idx))): sampled posterior with the correct normalisation
        truths (np.ndarray, shape=(len(which_idx),): true values for the selected run
    """
    which_idx, _, _ = select_pars(bounds, which_use, normalise=normalise)

    run_idx = np.where(test.index == run_number)[0]
    if run_idx.size == 0:
        raise IndexError(f'run number {run_number} not found in the dataset')
    run_idx = run_idx[0]

    if verb:    print(f'Sampling {n_samples} for run number {run_number}')

    try:
        x_obs = test.x_pca[run_idx]
    except:
        x_obs = test.x[run_idx]
    truths = test.theta[run_idx, which_idx].cpu().numpy()
    if verb:    print('True values', truths)

    posterior.to('cpu')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        samples = posterior.sample((n_samples,), x=x_obs, show_progress_bars=verb).cpu().numpy()

    if normalise:
        restored_samples = np.zeros_like(samples)
        restored_truths = np.zeros_like(truths)
        for i, param in enumerate(which_use):
            restored_samples[:,i] = denormalize_param(samples[:,i], param, bounds)
            restored_truths[i] = denormalize_param(truths[i], param, bounds)
        if verb:    print('Restored initial normalisation of the labels')

        return restored_samples, restored_truths
    else:
        return samples, truths

