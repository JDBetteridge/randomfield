from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt


def plot_estimator(iterations, cumean, mean=None, labels=None, ax=plt):
    if mean is None:
        mean = cumean[:, -1]
    ax.plot(iterations, cumean.T)
    ax.set_prop_cycle(None)
    ax.plot([0, iterations[-1]], [mean, mean], ls='--', lw=1)
    for ii, val in enumerate(mean):
        pos = (iterations[-1], val)
        offset = (3, 0)
        ax.annotate(
                f'{val:6.4f}',
                xy=pos, xycoords='data',
                xytext=offset, textcoords='offset points',
                fontsize='medium', color=f'C{ii}', ha='left'
            )
    if labels is not None:
        ax.legend(labels, title='Mesh Size')
    ax.set_xlabel('Samples, n')
    ax.set_ylabel(r'$\frac{1}{n}\sum_{i=1}^n || u_i ||_{{L^2}}^2$')
    ax.set_title('Convergence of estimators to expectation')
    return ax


def plot_distribution(raw_data, mean, labels, axes=None):
    # Note: Normally generates a figure rather than take an axis as arg

    if axes is None:
        k = raw_data.shape[0]
        h = int(np.floor(np.sqrt(k)))
        w = int(np.ceil(k/h))
        fig, axes = plt.subplots(h, w, sharex=True, sharey=True)

    for ii, (data, mean, size, ax) in enumerate(zip(raw_data, mean, labels, axes.ravel())):
        ax.hist(data, bins=20, density=True, color=f'C{ii}')
        ax.vlines([mean], 0, 1, transform=ax.get_xaxis_transform(), color='k')
        pos = (mean, 1.5)
        offset = (20, 0)
        ax.annotate(
                f'Mean :\n$\hat{{\mu}}=${mean:6.4f}',
                xy=pos, xycoords='data',
                xytext=offset, textcoords='offset points',
                fontsize='medium', color='k', ha='left'
            )
        ax.set_title(str(size))

    if axes is not None:
        fig = axes.ravel()[0].figure
        fig.suptitle('Histograms of L2 norm')

    return fig, axes


def plot_MC_bounds(iterations, cumean, var, color='C0', labels=None, ax=plt):
    burn_in = int(0.05*iterations.size)
    label_coord = -int(0.5*iterations.size)
    ax.plot(iterations, np.abs(cumean - 1), color=color)
    for ii in range(1, 4):
        ax.plot(iterations[burn_in:], +ii*np.sqrt(var/iterations[burn_in:]), color='k', ls='--', lw=1)
    pos = (iterations[label_coord], 3*np.sqrt(var/iterations[label_coord]))
    offset = (3, 10)
    ax.annotate(
            f'Variance: \n$\hat{{\sigma}} = $ {var:6.4f}\nConfidence intervals\n$k \hat{{\sigma}}/\sqrt{{n}}$',
            xy=pos, xycoords='data',
            xytext=offset, textcoords='offset points',
            fontsize='medium', color='k', ha='left'
        )
    if labels is not None:
        ax.legend([labels], title='Mesh Size')
    ax.set_xlabel('Samples, n')
    ax.set_ylabel(r'abs$\left( \frac{1}{n}\sum_{i=1}^n || u_i ||_{{L^2}}^2 - \sigma^2 |G|\right)$')
    ax.set_title('Monte Carlo convergence for finest mesh\n with confidence intervals')
    return ax


def plot_MC_rate(iterations, cumean, var, color='C0', labels=None, ax=plt):
    label_coord = -int(0.8*iterations.size)
    ax.loglog(iterations, np.abs(cumean - 1), color=color)
    endpoints = [iterations[0], iterations[-1]]
    for ii in range(1, 4):
        ax.loglog(endpoints, ii*np.sqrt(var/endpoints), color='k', ls='--', lw=1)
    it = iterations[label_coord]
    pos = (it, 3*np.sqrt(var/it))
    offset = (3, 0)
    ax.annotate(
            r'$k \hat{{\sigma}}/\sqrt{{n}}$',
            xy=pos, xycoords='data',
            xytext=offset, textcoords='offset points',
            fontsize='medium', color='k', ha='left'
        )
    if labels is not None:
        ax.legend([labels], title='Mesh Size')
    ax.set_xlabel('Samples, n')
    ax.set_ylabel(r'abs$\left( \frac{1}{n}\sum_{i=1}^n || u_i ||_{{L^2}}^2 - \sigma^2 |G|\right)$')
    ax.set_title('Monte Carlo convergence for finest mesh\n with confidence intervals')
    return ax


def plot_h_rate(mesh_size, mean, var, iters=None, ax=plt):
    # ~ ax5.loglog(mesh_size, np.abs(1 - P_mean))
    # ~ ax5.loglog(mesh_size, 1024*np.array(mesh_size, dtype='float64')**(-2))
    ax.plot(mesh_size, np.abs(1 - mean))

    endpoints = np.array([mesh_size[0], mesh_size[-1]], dtype='float64')
    reference = 1024*endpoints**(-2)
    ax.plot(endpoints, reference, 'k')
    if iters is not None:
        for ii in range(1, 4):
            ax.plot(mesh_size, 1 - mean + ii*np.sqrt(var/iters), color='k', ls='--', lw=1)
            ax.plot(mesh_size, 1 - mean - ii*np.sqrt(var/iters), color='k', ls='--', lw=1)
    # ~ ax5.boxplot(
        # ~ 1 - Pl.T[:,:maxi],
        # ~ sym='b+',
        # ~ positions=mesh_size,
        # ~ widths=np.array(mesh_size)/4,
        # ~ showmeans=True,
        # ~ meanprops={'marker': 'x', 'mec': 'r'}
        # ~ )
    # ~ parts = ax5.violinplot(
                # ~ 1 - Pl.T,
                # ~ positions=mesh_size,
                # ~ widths=np.array(mesh_size)/4,
                # ~ showmeans=True,
                # ~ showmedians=True,
                # ~ )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim([1e-4, 10])
    return ax

def _main(args):
    factor = 2
    data = np.load(args.data, allow_pickle=True)
    points = data['points'][()]
    point_vals = data['point_vals'][()]
    L2 = data['L2'][()]
    volumes = data['volumes'][()]

    mesh_size = [k for k in L2.keys()]
    L2 = np.array([L2[s] for s in mesh_size])
    iterations = np.arange(1, L2.shape[1]+1)
    ones = np.ones_like(L2[0])
    vols = np.array([volumes[s] for s in mesh_size])
    mesh_size = np.array([m/factor for m in mesh_size], dtype='int32')

    # In Monte Carlo notation the L2 norms we have found correspond
    # to the variable P_l
    Pl = L2

    P_mean = np.mean(Pl, axis=1)
    P_var = np.var(Pl, axis=1)
    P_cumean = np.cumsum(Pl, axis=1)/iterations
    P_cuvar = np.cumsum((Pl - P_cumean)**2, axis=1)/iterations
    P_cuCI = 3*np.sqrt(P_cuvar/iterations)

    analytic = np.abs(P_cumean - args.variance*np.outer(vols, ones))
    approx = np.abs(P_cumean - np.outer(P_mean, ones))
    approx[:, -1] = np.nan

    # Print statistics:
    print('P_l statistics')
    print('Level      : ', ' & '.join(f'{s:6d}' for s in mesh_size))
    print('Mean       : ', ' & '.join(f'{s:6.4f}' for s in P_mean))
    print('Variance   : ', ' & '.join(f'{s:6.4f}' for s in P_var))

    # Plot estimator for expectation
    fig1, ax1 = plt.subplots(1, 1)
    fig1.set_size_inches(6, 6)
    plot_estimator(
        iterations,
        P_cumean,
        labels=mesh_size,
        ax=ax1
        )
    pngname = args.prefix + 'convergence_plot1.png'
    fig1.savefig(pngname, bbox_inches='tight', dpi=300)

    # Plot distribution of L2 norms
    fig2, ax2 = plot_distribution(Pl, P_mean, mesh_size, axes=None)
    fig2.set_size_inches(12, 6)
    pngname = args.prefix + 'L2histogram_plot2.png'
    fig2.savefig(pngname, bbox_inches='tight', dpi=300)

    # Plot MC bounds for finest mesh
    fig3, ax3 = plt.subplots(1, 1)
    plot_MC_bounds(iterations, P_cumean[-1], P_var[-1], color='C6', ax=ax3)
    pngname = args.prefix + 'MC_CIbounds_plot3.png'
    fig3.savefig(pngname, bbox_inches='tight', dpi=300)

    # Plot MC convergence rate for finest mesh
    fig4, ax4 = plt.subplots(1, 1)
    fig4.set_size_inches(6, 6)
    plot_MC_rate(iterations, P_cumean[-1], P_var[-1], color='C6', ax=ax4)
    pngname = args.prefix + 'MCrate_plot4.png'
    fig4.savefig(pngname, bbox_inches='tight', dpi=300)

    # Plot FE convergence (convergence in h)
    fig5, ax5 = plt.subplots(1, 1)
    fig5.set_size_inches(6, 6)
    plot_h_rate(mesh_size, P_mean, P_var, iters=iterations[-1], ax=ax5)
    pngname = args.prefix + 'hrate_plot5.png'
    fig5.savefig(pngname, bbox_inches='tight', dpi=300)


parser = ArgumentParser()
parser.add_argument('--data',
                    default=None,
                    type=str,
                    help='filename.npz from a previous run')
parser.add_argument('--prefix',
                    default='',
                    type=str,
                    help='filename.npz from a previous run')
parser.add_argument('--variance',
                    default=1,
                    type=float,
                    help='Variance (probably don\'t change this)')

if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    _ = _main(args)
