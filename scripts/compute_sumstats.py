"""
This file is used to read the observed richness and compute 
different summary statistics that are saved on disk

Expected structure:
    - read
    {PP_DIR}/{RUN_BASENAME}_{ID}/{_DEF_DIRNAME_RICH}/
    {RUNS_DIR}/{RUN_BASENAME}_{ID}/

    - save
    {STATS_DIR}/{sum-stat specifics}/{RUN_BASENAME}_{ID}.npz

"""
import argparse
import numpy as np
import glob
import os 
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import time
from datetime import datetime, timedelta
from tqdm import tqdm

import pinsbi.summarystats as ss

# Defaults
_DEF_BASENAME = 'model'
_DEF_MTHR = 1e13
_DEF_MASSKNEE = 5e14
_DEF_MASSMAX = 1e16
_DEF_MASS_BINS = 5
_DEF_ZMIN = 0.2
_DEF_ZKNEE = 1.5
_DEF_ZMAX = 1.
_DEF_Z_BINS = 4
_DEF_LMBMIN = 30.
_DEF_LMBKNEE = 150.
_DEF_LMBMAX = 300.
_DEF_LMB_BINS = 4

_DEF_DIRNAME_RICH = 'richness_global'
_DEF_BASEFILENAME_RICH = 'halo_richness_merged_zmax'
_DEF_EXTENSION_RICH = '.npz'

@dataclass
class ConfigArgs:
    PP_DIR: str
    RUNS_DIR: str
    model_basename: str
    mode: str
    z_max_shell: float
    mass_thr: float
    save_dir: str

def printlog(text: str):
    now = datetime.now().strftime('%H:%M:%S')
    print(f'\033[92m[SUMSTATS: {now}]>\033[0m ' + text)

def format_timedelta_dhms(diff):
    total_s = int(diff.total_seconds())
    days = total_s // 86400
    res = total_s % 86400
    return f'{days}-{time.strftime("%H:%M:%S", time.gmtime(res))}'

def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        printlog(f'Created directory: {path}')

# ======================================================================================== #
# Reading
# ======================================================================================== #

def read_richness_file(path: str):
    """
    Reads from disc the output of the CosmoPostProcess richness module

    Args:
        path (str): path to the richness file

    Returns:
        Z, MASS, LAMBDA_TRUE, LAMBDA_OBS 
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f'File {path} not found')

    data = np.load(path)

    if args.m_threshold < data['m_threshold']:
        raise ValueError(f'The provided mass threshold ({args.m_threshold}) is smaller than the one used to compute the richness ({data["m_threshold"]}).')

    Z = data['z']
    MASS = data['Mass']
    LAMBDA_TRUE = data['lambda_true']
    LAMBDA_OBS = data['lambda_obs']

    return Z, MASS, LAMBDA_TRUE, LAMBDA_OBS

def read_zshells_file(run_path: str):
    """
    Reads from disc the redshift shells
    """
    raise NotImplementedError

def read_plc_file(run_path: str, model: str):
    """
    Reads from disc the Pinocchio PLC

    Args:
        run_path (str): path to the run
        model (str): run flag

    Returns:
        Ztrue, Z, MASS
    """
    from ReadPinocchio5 import plc as pin_plc

    path = f'{run_path}/pinocchio.{model}.plc.out'

    if not os.path.isfile(path):
        raise FileNotFoundError(f'File {path} not found')

    plc = pin_plc(path, silent=True)

    Ztrue = plc.data['truez']
    Z = plc.data['obsz']
    MASS = plc.data['Mass']

    return Ztrue, Z, MASS

def read_masscal_file(run_path: str, model: str):
    """
    Reads from disc the mass calibrated Pinocchio PLC .fits file

    Args:
        run_path (str): path to the run
        model (str): run flag

    Returns:
        Ztrue, Z, MASS
    """
    from astropy.io import fits

    path = f'{run_path}/pinocchio.{model}.plc.masscalib.fits'

    if not os.path.isfile(path):
        raise FileNotFoundError(f'File {path} not found')

    plc = fits.getdata(path)

    Ztrue = plc['truez']
    Z = plc['obsz']
    MASS = plc['Mass']

    return Ztrue, Z, MASS

def read_cat_file(run_path: str, model: str, redshift: float):
    """
    Reads from disc the Pinocchio catalog

    Args:
        run_path (str): path to the run
        model (str): run flag
        redshift (float): catalog redshift

    Returns:
        POS, MASS
    """
    from pinsbi.ReadPinocchio5 import catalog as pin_cat

    path = f'{run_path}/pinocchio.{redshift:.4f}.{model}.catalog.out'

    if not os.path.isfile(path):
        raise FileNotFoundError(f'File {path} not found')

    cat = pin_cat(path, silent=True)

    POS = cat.data['pos']
    MASS = cat.data['Mass']

    return POS, MASS

# ======================================================================================== #
# Summ. stats single process
# ======================================================================================== #

def _process_single(args_tuple):
    """
    A single worker function

    Args:
        args_tuple (tuple): containing, in order: i, ConfigArgs, z_edges, lmb_edges, mass_edges 
    """
    i, args, z_edges, lmb_edges, mass_edges = args_tuple

    model_name = args.model_basename + f'_{i}'

    try:

        if args.mode == 'richness':

            richness_dir = f'{args.PP_DIR}/{model_name}/{_DEF_DIRNAME_RICH}/'
            richness_file = f'{richness_dir}/{_DEF_BASEFILENAME_RICH}{args.z_max_shell}{_DEF_EXTENSION_RICH}'
            Z, MASS, LAMBDA_TRUE, LAMBDA_OBS = read_richness_file(richness_file)

            # Apply mass selection
            mask = MASS > args.mass_thr
            Z, MASS, LAMBDA_TRUE, LAMBDA_OBS = Z[mask], MASS[mask], LAMBDA_TRUE[mask], LAMBDA_OBS[mask]

            # Number counts in redshift bins
            counts_in_z = ss.counts(Z, z_edges)
            # Number counts in richness bins
            counts_in_lmb = ss.counts(LAMBDA_OBS, lmb_edges)
            # Number counts in richness and redshift bins
            counts_in_lmb_z = ss.counts_in_2dim(LAMBDA_OBS, Z, lmb_edges, z_edges)
            # Number counts in mass and redshift bins
            counts_in_mass_z = ss.counts_in_2dim(MASS, Z, mass_edges, z_edges)
            # Richness in redshift bins
            lmb_in_z = ss.lmb_in_z(LAMBDA_OBS, Z, z_edges)

            save_path = f'{args.save_dir}/{model_name}.npz'
            save_data = {
                    'counts_z': counts_in_z,
                    'counts_lmb': counts_in_lmb,
                    'counts_lmb_z': counts_in_lmb_z,
                    'counts_mass_z': counts_in_mass_z,
                    'lmb_z': lmb_in_z,
                    'z_edges': z_edges,
                    'lmb_edges': lmb_edges,
                    'mass_edges': mass_edges,
                    }
            np.savez(save_path, **save_data)

        elif args.mode in ['plc', 'plc_masscalib']:

            run_dir = f'{args.RUNS_DIR}/{model_name}/'
            if args.mode == 'plc':
                Ztrue, Z, MASS = read_plc_file(run_dir, model_name)
            elif args.mode == 'plc_masscalib':
                Ztrue, Z, MASS = read_masscal_file(run_dir, model_name)

            # Apply mass selection
            mask = MASS > args.mass_thr
            Ztrue, Z, MASS = Ztrue[mask], Z[mask], MASS[mask]

            # Number counts in redshift bins
            counts_in_z = ss.counts(Ztrue, z_edges)
            # Number counts in mass and redshift bins
            counts_in_mass_z = ss.counts_in_2dim(MASS, Ztrue, mass_edges, z_edges)

            save_path = f'{args.save_dir}/{model_name}.npz'
            save_data = {
                    'counts_z': counts_in_z,
                    'counts_mass_z': counts_in_mass_z,
                    'z_edges': z_edges,
                    'mass_edges': mass_edges,
                    }
            np.savez(save_path, **save_data)

        elif arsg.mode == 'zshells':

            run_dir = f'{args.RUNS_DIR}/{model_name}/'

            # TODO: read each zshell file (add def pattern, use glob to find the names), count the elements inside it and store them to file
            raise NotImplementedError

        return i, None # None = success

    except Exception as e:

        return i, e # Report the error without killing the process

if __name__ == '__main__':

    """
    Fundamental assumptions to make this properly work:
        1. All models are stored in the PostProc dir as `{base-name}_{i}` as done by the PinocchioSBI-gen_mocks routines
        2. All Pinocchio runs are stored in runs dir as `{Base-name}_{i} as well
        3. The richness basename is directly followed by the z max, e.g.: `{_DEF_BASEFILENAME_RICH}{ZMAX}{_DEF_EXTENSION_RICH}`

    """
    init_time = time.time()

    # ======================================================================================== #
    # Main setup
    # ======================================================================================== #

    parser = argparse.ArgumentParser(description='Compute different summary statistics for the post-processed Pinocchio dataset')

    parser.add_argument('--runs-dir', type=str, required=True)
    parser.add_argument('--postproc-dir', type=str, required=True)
    parser.add_argument('--stats-dir', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--total-runs', type=int, required=False, default=None)
    parser.add_argument('--base-name', type=str, default=_DEF_BASENAME)

    # Binning edges
    parser.add_argument('--z-min-shell', type=float, default=_DEF_ZMIN)
    parser.add_argument('--z-knee', type=float, default=_DEF_ZKNEE)
    parser.add_argument('--z-max-shell', type=float, default=_DEF_ZMAX)
    parser.add_argument('--m-threshold', type=float, default=_DEF_MTHR)
    parser.add_argument('--mass-knee', type=float, default=_DEF_MASSKNEE)
    parser.add_argument('--mass-max', type=float, default=_DEF_MASSMAX)
    parser.add_argument('--lmb-min', type=float, default=_DEF_LMBMIN)
    parser.add_argument('--lmb-knee', type=float, default=_DEF_LMBKNEE)
    parser.add_argument('--lmb-max', type=float, default=_DEF_LMBMAX)

    # Richness specific
    parser.add_argument('--dirname-richness', type=str, default=_DEF_DIRNAME_RICH)
    parser.add_argument('--basename-richness', type=str, default=_DEF_BASEFILENAME_RICH)
    parser.add_argument('--extension-richness', type=str, default=_DEF_EXTENSION_RICH)

    # Binning
    parser.add_argument('--z-bins', type=int, default=_DEF_Z_BINS)
    parser.add_argument('--lmb-bins', type=int, default=_DEF_LMB_BINS)
    parser.add_argument('--mass-bins', type=int, default=_DEF_MASS_BINS)
    parser.add_argument('--log-z', action='store_true')
    parser.add_argument('--log-mass', action='store_true')
    parser.add_argument('--log-lmb', action='store_true')
    parser.add_argument('--use-zknee', action='store_true')
    parser.add_argument('--use-massknee', action='store_true')
    parser.add_argument('--use-lmbknee', action='store_true')

    parser.add_argument('--n-workers', type=int, default=None)

    args = parser.parse_args()

    accepted_modes = ['richness', 'zshells', 'plc', 'plc_masscalib']
    assert args.mode in accepted_modes, f'{args.mode} is not a valid mode. Accepted {accepted_modes}.'

    # ======================================================================================== #

    RUNS_DIR = args.runs_dir
    PP_DIR = args.postproc_dir
    ST_DIR = args.stats_dir
    model_basename = args.base_name

    which_dir = PP_DIR if args.mode == 'richness' else RUNS_DIR
    if args.total_runs is None:

        if not os.path.isdir(which_dir):
            raise NotADirectoryError(f'Directory {which_dir} not found.')

        N = sum(1 for entry in os.scandir(which_dir) if entry.is_dir() and entry.name.startswith(f'{model_basename}_'))

    else:
        N = args.total_runs

    if args.n_workers is not None:
        n_workers = args.n_workers
    elif 'SLURM_CPUS_PER_TASK' in os.environ:
        n_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        n_workers = os.cpu_count()

    # ======================================================================================== #

    printlog(f'Computing summary statistics for {model_basename}_0-{N - 1}. Mode = {args.mode}, {n_workers} workers')

    # Z edges
    if args.log_z:

        if args.use_zknee:
            printlog(f'Using variable logarithmic spacing: redshift bins={args.z_bins}')
            z_edges = np.append(np.logspace(np.log10(args.z_min_shell), np.log10(args.z_knee), args.z_bins), args.z_max_shell)
        else:
            printlog(f'Using logarithmic spacing: redshift bins={args.z_bins}')
            z_edges = np.logspace(np.log10(args.z_min_shell), np.log10(args.z_max_shell), args.z_bins+1)

        z_spacing = 'logvar' if args.use_zknee else 'log'

    else:
        
        if args.use_zknee:
            printlog(f'Using variable linear spacing: redshift bins={args.z_bins}')
            z_edges = np.append(np.linspace(args.z_min_shell, args.z_knee, args.z_bins), args.z_max_shell)
        else:
            printlog(f'Using linear spacing: redshift bins={args.z_bins}')
            z_edges = np.linspace(args.z_min_shell, args.z_max_shell, args.z_bins+1)

        z_spacing = 'linvar' if args.use_zknee else 'lin'

    # Richness edges
    if args.log_lmb:

        if args.use_lmbknee:
            printlog(f'Using variable logarithmic spacing: richness bins={args.lmb_bins}')
            lmb_edges = np.append(np.logspace(np.log10(args.lmb_min), np.log10(args.lmb_knee), args.lmb_bins), args.lmb_max)
        else:
            printlog(f'Using logarithmic spacing: richness bins={args.lmb_bins}')
            lmb_edges = np.logspace(np.log10(args.lmb_min), np.log10(args.lmb_max), args.lmb_bins+1)

        lmb_spacing = 'logvar' if args.use_lmbknee else 'log'

    else:

        if args.use_lmbknee:
            printlog(f'Using variable linear spacing: richness bins={args.lmb_bins}')
            lmb_edges = np.append(np.linspace(args.lmb_min, args.lmb_knee, args.lmb_bins), args.lmb_max)
        else:
            printlog(f'Using linear spacing: richness bins={args.lmb_bins}')
            lmb_edges = np.linspace(args.lmb_min, args.lmb_max, args.lmb_bins+1)

        lmb_spacing = 'linvar' if args.use_lmbknee else 'lin'

    # Mass edges
    if args.log_mass:
        
        if args.use_massknee:
            printlog(f'Using variable logarithmic spacing: mass bins={args.mass_bins}')
            mass_edges = np.append(np.logspace(np.log10(args.m_threshold), np.log10(args.mass_knee), args.mass_bins), args.mass_max)
        else:
            printlog(f'Using logarithmic spacing: mass bins={args.mass_bins}')
            mass_edges = np.logspace(np.log10(args.m_threshold), np.log10(args.mass_max), args.mass_bins+1)

        mass_spacing = 'logvar' if args.use_massknee else 'log'

    else:

        if args.use_massknee:
            printlog(f'Using variable linear spacing: mass bins={args.mass_bins}')
            mass_edges = np.append(np.linspace(args.m_threshold, args.mass_knee, args.mass_bins), args.mass_max)
        else:
            printlog(f'Using linear spacing: mass bins={args.mass_bins}')
            mass_edges = np.linspace(args.m_threshold, args.mass_max, args.mass_bins+1)

        mass_spacing = 'linvar' if args.use_massknee else 'lin'

    # ======================================================================================== #
    # Directory structure
    # ======================================================================================== #

    save_dir = f'{ST_DIR}/{args.mode}_m{args.m_threshold:.1e}'.replace('+','') + f'_z{args.z_max_shell:.3f}-{args.z_min_shell:.3f}'
    save_dir += f'_{z_spacing}-zbins{args.z_bins}_{mass_spacing}-massbins{args.mass_bins}'
    if args.mode == 'richness': save_dir += f'_{lmb_spacing}-lmbbins{args.lmb_bins}'

    create_dir(ST_DIR)
    create_dir(save_dir)

    printlog(f'Saving stats in path: {save_dir}')

    # ======================================================================================== #
    # Loop over each catalogue with independent processes
    # ======================================================================================== #

    # another Args class, specifically to pass the values to each process
    config = ConfigArgs(RUNS_DIR = RUNS_DIR, PP_DIR = PP_DIR,
                        model_basename = model_basename, mode = args.mode,
                        z_max_shell = args.z_max_shell, mass_thr = args.m_threshold,
                        save_dir = save_dir
                        )

    # one tuple per run
    worker_args = [ (i, config, z_edges, lmb_edges, mass_edges) for i in range(N)]

    # CPU-bounded task, so we spawn indep. child processes (each with its own interpreter)
    failed = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:

        # associate a process (that runs _process_single with args "a") with the index i
        futures = {executor.submit(_process_single, a): a[0] for a in worker_args}
        for future in tqdm(as_completed(futures), total=N, desc='Computing summary stats'):
            i, err = future.result()

            if err is not None:
                printlog(f'WARNING: run {i} failed: {err}')
                failed.append(i)

    if failed:
        printlog(f'Failed runs: {failed} (total {len(failed)})')

    final_time = time.time() - init_time
    printlog(f'Finished. Elapsed wall-clock time: {str(timedelta(seconds=int(final_time)))}')
