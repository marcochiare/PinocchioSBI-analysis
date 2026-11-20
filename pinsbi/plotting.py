"""
This file contains usefull and recurrent functions used for plotting
"""
import numpy as np
import matplotlib.pyplot as plt

def lower_triangle_axes(labels, limits=None, size=2.4, space=0.05):
    """
    Empty lower-triangular grid for len(labels) parameters, diagonal excluded.
 
    Args:
        labels: list of labels to display.
        limits (default=None): limits for each panel.
        size (default=2.4): single panel size.
        space (default=0.05): spacing between the panels.

    Returns:
        fig, axes : axes is an (n-1, n-1) object array. axes[r, c] shows
            parameter c on x and parameter r+1 on y, and exists only for c <= r.
            Upper-triangle slots are None.

    Example:
        >>> keys = list(labels_dict.keys())
        >>> labels = [labels_dict[k] for k in keys]
        >>> n = len(keys)
        >>> fig, ax = lower_triangle_axes(labels, limits=None)
        >>> for r in range(n - 1):
        >>>     for c in range(r + 1):
        >>>         ax[r, c].scatter(allvals[keys[c]], allvals[keys[r + 1]], s=1, alpha=0.6)
    """
    n = len(labels)
    m = n - 1
    fig, grid = plt.subplots(m, m, sharex='col', sharey='row',
                             figsize=(size * m, size * m),
                             gridspec_kw=dict(wspace=space, hspace=space))
    grid = np.atleast_2d(grid)
    axes = np.empty((m, m), dtype=object)
 
    for r in range(m):
        for c in range(m):
            ax = grid[r, c]
            if c > r:                      # upper triangle: hide (don't remove, axes are shared)
                ax.set_visible(False)
                continue
            axes[r, c] = ax
            if limits is not None:
                ax.set_xlim(limits[c])
                ax.set_ylim(limits[r + 1])
            ax.tick_params(direction='in', top=True, right=True, labelrotation=45)
            if c == 0:
                ax.set_ylabel(labels[r + 1])
            if r == m - 1:
                ax.set_xlabel(labels[c])
            ax.label_outer()               # tick labels only on left column / bottom row
    return fig, axes

def plot_correlation_offdiag(corrs, corrs_std, lbi_corr, param_labels, legend_labels=None, fmt='.'):
    """
    Plot the off-diagonal correlation matrix elements difference between a list of correlation matrices
    and the LBI correlation matrix.
    """
    data = [np.asarray(d) for d in corrs]
    data_std = [np.asarray(d) for d in corrs_std]

    N = len(param_labels)
    ndata = len(data)

    if ndata == 1 :#and isinstance(legend_labels, str):
        legend_labels = [legend_labels]
    elif ndata > 1 and legend_labels is None:
        raise ValueError('Please provide the legend names for multiple distributions of data')
    elif ndata > 1:
        if len(legend_labels) != ndata:
            raise ValueError(f'legend_labels has length {len(legend_labels)}; expected {ndata} to match the number of distributions')

    lab_matr = np.array([[rf'{i}$-${j}' for j in param_labels] for i in param_labels])

    f, ax = plt.subplots(1,1, figsize=(6,3), layout='constrained')
    CCYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color'] 

    for idata, (corr, corr_std) in enumerate(zip(data, data_std)):
        color = CCYCLE[idata]
        first = True
        print(f'determinant {np.linalg.det(corr):.3f} ({legend_labels[idata]})')
        for i in range(N):
            for j in range(N):
                if i >= j: continue
                ax.errorbar(x=lab_matr[i,j], y=(corr[i,j]-lbi_corr[i,j]), yerr=corr_std[i,j], label=legend_labels[idata] if first else None,
                            fmt=fmt, c=color)
                first = False

    if legend_labels is not None:
        ax.legend()
    ax.axhline(0, c='k', linestyle='dashed')
    ax.set_ylabel(r'$\rho_{\rm SBI} - \rho_{\rm LBI}$')
    ax.set_title('Off-diagonal correlation')

    return f, ax
    
def plot_correlation_offdiag_vert(corrs, corrs_std, lbi_corr, param_labels, legend_labels=None, fmt='.'):
    """
    Plot the off-diagonal correlation matrix elements difference between a list of correlation matrices
    and the LBI correlation matrix.
    """
    data = [np.asarray(d) for d in corrs]
    data_std = [np.asarray(d) for d in corrs_std]
    N = len(param_labels)
    ndata = len(data)

    if ndata == 1 and isinstance(legend_labels, str):
        legend_labels = [legend_labels]
    elif ndata > 1 and legend_labels is None:
        raise ValueError('Please provide the legend names for multiple distributions of data')
    elif ndata > 1 and len(legend_labels) != ndata:
        raise ValueError(f'legend_labels has length {len(legend_labels)}; expected {ndata}')

    lab_matr = np.array([[rf'{i}$-${j}' for j in param_labels] for i in param_labels])

    # collect off-diagonal pair labels and indices
    pair_labels = []
    pair_idx = []
    for i in range(N):
        for j in range(N):
            if i >= j: continue
            pair_labels.append(lab_matr[i, j])
            pair_idx.append((i, j))

    npairs = len(pair_labels)
    yticks = np.arange(npairs)

    # vertical offsets so different corrs don't overlap
    offsets = np.linspace(-0.2, 0.2, ndata) if ndata > 1 else [0.0]

    f, ax = plt.subplots(1, 1, figsize=(6, npairs * 0.6 + 1), layout='constrained')
    CCYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idata, (corr, corr_std) in enumerate(zip(data, data_std)):
        color = CCYCLE[idata]
        first = True
        for k, (i, j) in enumerate(pair_idx):
            ax.errorbar(x = corr[i, j] - lbi_corr[i, j], y = yticks[k] - offsets[idata], xerr = corr_std[i, j],
                        label = legend_labels[idata] if first else None, fmt = fmt, c = color,
                        )
            first = False

    ax.set_yticks(yticks)
    ax.set_yticklabels(pair_labels)
    ax.axvline(0, c='k', linestyle='dashed')
    ax.set_xlabel(r'$\rho_{\rm SBI} - \rho_{\rm LBI}$')
    ax.set_title('Off-diagonal correlation')

    if legend_labels is not None:
        f.legend(loc='outside center right')

    return f, ax 

def plot_coverage_generic(alphas, marginal_coverage, joint_coverage,
                          n_test, param_names=None,
                          save_fig: bool = False, save_name: str = 'coverage'):
    """
    Plot a TARP test for each parameter and for the joint HPD

    This function expects the results from `compute_coverage_generic`
    """
    D = marginal_coverage.shape[1]
    if param_names is None:
        param_names = [rf'$\theta_{d}$' for d in range(D)]

    sigma = np.sqrt(alphas * (1 - alphas) / n_test)

    fig, axes = plt.subplots(1, D + 1, figsize=(4*(D+1), 4))

    for d, ax in enumerate(axes[:-1]):
        ax.plot([0,1],[0,1],'k--',lw=1)
        ax.fill_between(alphas, alphas-2*sigma, alphas+2*sigma,
                        alpha=0.2, color='C0')
        ax.plot(alphas, marginal_coverage[:, d], 'o-', color='C0')
        ax.set_title(f'Marginal: {param_names[d]}')
        ax.set_xlabel(r'Credibility Level $\alpha$')
        ax.set_ylabel('Expected Coverage')
        ax.set_aspect('equal')

    ax = axes[-1]
    ax.plot([0,1],[0,1],'k--',lw=1)
    ax.fill_between(alphas, alphas-2*sigma, alphas+2*sigma,
                    alpha=0.2, color='C1')
    ax.plot(alphas, joint_coverage, 'o-', color='C1')
    ax.set_title('Joint HPD coverage')
    ax.set_xlabel(r'Credibility Level $\alpha$')
    ax.set_ylabel('Expected Coverage')
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_fig:
        plt.savefig(f'{save_name}.png', dpi=500, bbox_inches='tight')
        plt.savefig(f'{save_name}.pdf', bbox_inches='tight')
        print(f'Plot saved as {save_name}.pdf(.png)')

    plt.show()

    return fig, axes

def plot_zscore_summary(results, param_names,
                        save_fig: bool = False, save_name: str = 'zscore_summary'):
    """
    Visual summary: z-scores and width ratios per parameter.
    """
    import matplotlib.pyplot as plt
    zs = [results[p]['z'] for p in param_names]
    wrs = [results[p]['width_ratio'] for p in param_names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))

    ax = axes[0]
    ax.barh(param_names, zs)
    ax.axvline(0, color='k', lw=1)
    ax.axvline(-1, color='gray', lw=1, ls='--', alpha=0.5)
    ax.axvline( 1, color='gray', lw=1, ls='--', alpha=0.5)
    ax.axvline(-2, color='red',  lw=1, ls='--', alpha=0.4)
    ax.axvline( 2, color='red',  lw=1, ls='--', alpha=0.4)
    ax.set_xlabel(r'$z$-score ')
    ax.set_title('Posterior mean tension')

    ax = axes[1]
    ax.barh(param_names, wrs)
    ax.axvline(1, color='k', lw=1, ls='--')
    ax.set_xlabel(r'$\sigma_{\rm NPE} / \sigma_{\rm LBI}$   ($>$1: NPE broader, $<1$: LBI broader)')
    ax.set_title('Posterior width ratio')

    plt.tight_layout()

    if save_fig:
        plt.savefig(f'{save_name}.png', dpi=500, bbox_inches='tight')
        plt.savefig(f'{save_name}.pdf', bbox_inches='tight')
        print(f'Plot saved as {save_name}.pdf(.png)')

    plt.show()
