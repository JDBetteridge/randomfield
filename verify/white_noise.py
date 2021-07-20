from firedrake import *

import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from matplotlib.patheffects import withStroke

def random_sample(V, samples=100):
    iterations = [1, 5] + [10*(ii + 1) for ii in range(samples//10)]
    u = TrialFunction(V)
    v = TestFunction(V)

    mass = inner(u, v)*dx
    mass_matrix = assemble(mass, mat_type='aij')

    M = np.array(mass_matrix.petscmat[:,:])

    C = np.zeros_like(M)
    expectationC = []
    expectationl2 = []
    suml2 = 0
    for ii in range(samples):
        white = WhiteNoise(V)
        C += np.outer(white.dat.data, white.dat.data)
        suml2 += white.dat.data
        if ii + 1 in iterations:
            expectationC.append(np.linalg.norm(M - (C/(ii+1))))
            expectationl2.append(np.linalg.norm(suml2/(ii+1)))
    C /= samples

    return M, C, iterations, (expectationl2, expectationC)

def plot_comparison(M, C, name=0):
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(9, 4)
    vmax = np.max([M, C])
    vmin = np.min([M, C])
    cb1 = ax[0].matshow(M)#, vmax=vmax, vmin=vmin)
    ax[0].set_title('Mass Matrix')
    fig.colorbar(cb1, ax=ax[0:2], location='bottom')
    ax[1].matshow(C, vmax=vmax, vmin=vmin)
    ax[1].set_title('Empirical Covariance')

    diff = M - C
    dvmax = np.max(np.abs(diff))
    cb2 = ax[2].matshow(diff, vmax=dvmax, vmin=-dvmax, cmap='coolwarm')
    ax[2].set_title('Difference')
    fig.colorbar(cb2, ax=[ax[2]], location='bottom', aspect=10)
    fig.savefig(f'white_noise_mat{name}.png', bbox_inches='tight', dpi=300)

def plot_convergence(iterations, expectation, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(9, 4)
    ax[0].loglog(iterations, expectation[0])
    ax[1].loglog(iterations, expectation[1])
    if ax is None:
        fig.savefig(f'white_noise_cvgce{iterations[-1]}.png', bbox_inches='tight', dpi=300)

def plot_guidelies(iterations, expectation, ax=None):
    if ax is not None:
        for axis, ex in zip(ax, expectation):
            axis.loglog(iterations, [1.5*(ii**(-0.5))*ex[0] for ii in iterations], 'k--')
            it = iterations[len(iterations)//2]
            pos = (it, 1.5*(it**(-0.5))*ex[0])
            anchor = 'left'
            offset = (3, 2)
            patheff = [withStroke(linewidth=2, foreground='w')]
            axis.annotate(
                '$n^{-1/2}$',
                 xy=pos, xycoords='data',
                 xytext=offset, textcoords='offset points',
                 fontsize='small', color='k', ha=anchor,
                 path_effects=patheff
            )

def _main(args):
    degrees = [d for d in range(args.basedeg, args.basedeg + args.degs)]
    if args.dim == 2:
        basemesh = UnitSquareMesh(args.baseN, args.baseN)
    elif args.dim == 3:
        basemesh = UnitCubeMesh(args.baseN, args.baseN, args.baseN)
    mh = MeshHierarchy(basemesh, args.levels-1)

    fig, ax = plt.subplots(args.degs, 2, sharex=True, sharey=True)
    if args.degs == 1:
        ax = ax.reshape((-1, 2))
    fig.set_size_inches(9, 4*args.degs)

    ax[0, 0].set_title('$||\mathbb{E}[b]||_{\ell^2}$')
    ax[0, 1].set_title('$||\mathbb{E}[M - bb^T]||_2$')
    for deg, rowax in zip(degrees, ax):
        for mesh in mh:
            V = FunctionSpace(mesh, 'CG', deg)
            M, C, iterations, expectation = random_sample(V, samples=args.samples)
            if args.baseN == 4 and args.levels == 1 and args.basedeg == 1 and args.degs == 1:
                plot_comparison(M, C, name=f'{args.dim}D_{args.baseN}_deg{deg}_{args.samples}')
            plot_convergence(iterations, expectation, rowax)
        rowax[0].set_ylabel(f'Degree {deg}')
        plot_guidelies(iterations, expectation, rowax)

    ax[-1, 0].set_xlabel('Samples')
    ax[-1, 1].set_xlabel('Samples')
    ax[-1, -1].legend([str(args.baseN*2**l) for l in range(args.levels)], title='Mesh size')
    fig.suptitle(f'White noise convergence in {args.dim}D')
    fig.savefig(f'{args.dim}D_white_noise_cvgce{iterations[-1]}.png', bbox_inches='tight', dpi=300)


parser = ArgumentParser()
parser.add_argument('--samples',
                    default=1000,
                    type=int,
                    help='Number of samples to draw')
parser.add_argument('--dim',
                    default=2,
                    type=int,
                    choices=[2, 3],
                    help='Dimension')
parser.add_argument('--baseN',
                    default=4,
                    type=int,
                    help='Smallest mesh size')
parser.add_argument('--levels',
                    default=3,
                    type=int,
                    help='Number of mesh refinements')
parser.add_argument('--basedeg',
                    default=1,
                    type=int,
                    help='Smallest degree (>=1)')
parser.add_argument('--degs',
                    default=3,
                    type=int,
                    help='Number of degrees to test')


if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    _ = _main(args)
