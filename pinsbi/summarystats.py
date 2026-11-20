"""
PINSBI: summarystats

This file provides a collection of hand-crafted summary statistics
which combine histograms in redshift/richness bins or both

The script provides the following stats:
    - cumul_counts
    - counts
    - counts_in_2dim
    - lmb_in_z
    - hmf

"""
import numpy as np

from scipy.stats import gaussian_kde

def cumul_counts(data, thresholds):
    """
    Compute the number of elements above each threshold
    """
    return np.array([
        np.count_nonzero(data > t) for t in thresholds
        ])

def counts(data, edges):
    """
    Compute the number of elements in bins (wrapper to np.histogram)

    Args:
        data: the data to put into the histogram
        edges: edges
    """
    return np.histogram(data, bins=edges)[0]

def counts_in_2dim(data_one, data_two, edges_one, edges_two):
    """
    Compute the number counts in bins of two different probes to return
    a two dimensional matrix (wrapper to np.histogram2d)

    E.g.: number counts in richness and redshift bins

    Args:
        data_one: catalogue corresponding to the first probe
        data_two: catalogue corresponding to the second probe
        edges_one: edges for the probe-one bins [probe1_i, probe1_(i+1))
        edges_two: edges for the probe-two bins [probe2_i, probe2_(i+1))

    Return:
        counts: number of elements in probe-one bins [probe1_i, probe1_(i+1)) and probe-two bins [probe2_i, probe2_(i+1))
    """
    return np.histogram2d(data_one, data_two, bins=[edges_one, edges_two])[0]

def lmb_in_z(data_rich, data_z, edges):
    """
    Compute the richness in redshift bins

    Args:
        data_rich: richness catalogue
        data_z: redshift catalogue, corresponding to the richness catalogue
        edges: list or array of redshift edges [z_i, z_(i+1))

    Return:
        rich_counts: richness in redshift bins [z_i, z_(i+1))
    """
    tot_rich_bins = np.zeros(len(edges) - 1)
    mean_rich_bins = np.zeros_like(tot_rich_bins)

    for i in range(len(edges) - 1):
        z_edge_min = edges[i]
        z_edge_max = edges[i+1]

        red_bin_mask = (data_z >= z_edge_min) * (data_z < z_edge_max)
        rich_in_bin = data_rich[red_bin_mask]

        tot_rich_bins[i] = rich_in_bin.sum()
        mean_rich_bins[i] = rich_in_bin.mean()

    return tot_rich_bins, mean_rich_bins

def hmf(mass, mass_values, box_side):
    """
    Compute the Halo Mass Function (HMF) using the given mass bins

    Args:
        mass: mass catalogue in Msun/h
        mass_values: centers of the mass bins in Msun/h
        box_side: side of the box in Mpc/h

    Return:
        hmf n(M): halo mass function in h^4 Mpc^-3 Msun^-1
    """

    edges = np.zeros(len(mass_values) + 1)
    edges[1:-1] = (mass_values[1:] + mass_values[:-1]) / 2

    edges[0] = mass_values[0] - (mass_values[1] - mass_values[0]) / 2
    edges[-1] = mass_values[-1] + (mass_values[-1] - mass_values[-2]) / 2

    counts, _ = np.histogram(mass, bins=edges)

    volume = box_side**3
    dM = np.diff(edges)

    return counts / (dM * volume)

def FoM(
        samples: np.ndarray,
        indices: list[int],
        credibility_level: float = 0.95,
        ):
    
    x = samples[:, indices[0]]
    y = samples[:, indices[1]]
    
    #nx, ny = 300, 300

    #kde = gaussian_kde(np.stack([x, y]))
    #
    #x_grid = np.linspace(x.min(), x.max(), nx)
    #y_grid = np.linspace(y.min(), y.max(), ny)
    #X, Y = np.meshgrid(x_grid, y_grid)
    #Z = kde(np.stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    #
    ##densities = kde(np.stack([x,y]))
    ##threshold = np.quantile(densities, 1 - credibility_level)
    #z_sorted = np.sort(Z.ravel())[::-1]
    #z_cumsum = np.cumsum(z_sorted) / np.sum(z_sorted)
    #threshold = z_sorted[np.searchsorted(z_cumsum, credibility_level)]
    #
    ## Area = number of grid points inside credible region * pixel area
    #dx = (x.max() - x.min()) / nx
    #dy = (y.max() - y.min()) / ny
    #area = np.sum(Z >= threshold) * dx * dy
    #
    #fom = 1.0 / area

    cov = np.cov(np.stack([x,y]))
    det = np.linalg.det(cov)


    return 1. / np.sqrt(np.abs(det))
