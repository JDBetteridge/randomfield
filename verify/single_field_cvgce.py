from argparse import ArgumentParser
# ~ from firedrake import *
from mpi4py import MPI
COMM_WORLD = MPI.COMM_WORLD
from math import sqrt, ceil
from scipy.special import gamma
from scipy.special import kv as bessel2

import numpy as np
import matplotlib.pyplot as plt

def true_covariance(r, sigma=1, nu=1, lambd=0.2):
    kappa=np.sqrt(8*nu)/lambd
    cov = sigma**2
    cov /= (2**(nu - 1))*gamma(nu)
    cov *= ((kappa*r)**nu)
    cov *= bessel2(nu, kappa*r)
    return cov

def indicator_f(f, mesh=None):
    if mesh is None:
        mesh = f.function_space().mesh()
    x = SpatialCoordinate(mesh)
    d = mesh.topological_dimension()
    interval1 = lambda z, a, b: conditional(And(a < z, z < b), 1, 0)
    intervalf = lambda z, a, b: conditional(And(a < z, z < b), f, 0)
    product = intervalf(x[0], 0, 1)
    for ii in range(1, d):
        product *= interval1(x[ii], 0, 1)
    return product

def _main(args):
    comm = COMM_WORLD
    args.variance = 1 # Maybe global ???

    if args.data:
        data = np.load(args.data, allow_pickle=True)
        iterations = data['iterations'][()]
        points = data['points'][()]
        point_vals = data['point_vals'][()]
        expectation = data['expectation'][()]
        volumes = data['volumes'][()]
    else:
        pcg = PCG64(seed=args.seed)
        rng = RandomGenerator(pcg)

        dim = args.dim
        mesh_size = [2**(r + 1) for r in range(15)]
        mesh_size = [m for m in mesh_size if m <= args.maxmesh]
        deg = args.deg
        iterations = [1, 5] + [10*(ii + 1) for ii in range(args.samples//10)]
        N = iterations[-1]
        VAR = 1
        smoothness = args.smoothness
        Npoints = 20
        if comm.rank == 0:
            points = rng.uniform(size=(Npoints, dim))
        else:
            points = np.zeros((Npoints, dim))
        comm.Bcast(points, root=0)
        CHOP = args.chop
        volumes = {}
        expectation = {}
        point_vals = {}
        for size in mesh_size:
            if dim == 2:
                if CHOP:
                    mesh = SquareMesh(size, size, 2)
                    mesh.coordinates.dat.data[:, :] -= 0.5
                else:
                    mesh = UnitSquareMesh(size, size)
            elif dim == 3:
                if CHOP:
                    mesh = CubeMesh(size, size, size, 2)
                    mesh.coordinates.dat.data[:, :] -= 0.5
                else:
                    mesh = UnitCubeMesh(size, size, size)
            V = FunctionSpace(mesh, 'CG', deg)
            GRF = GaussianRF(V, mu=0, sigma=VAR, smoothness=smoothness, correlation_length=0.2, rng=rng)

            sumL2 = 0
            point_vals[size] = np.zeros((N, Npoints))
            expectation[size] = []
            sample = indicator_f(Constant(1.0), mesh=mesh)
            volumes[size] = assemble(dot(sample, sample) * dx(domain=mesh))

            if comm.rank == 0:
                print('Mesh size:', size)
            for ii in range(N):
                sample = GRF.sample()
                point_vals[size][ii, :] = sample.at(points)
                sample = indicator_f(sample)
                sumL2 += assemble(dot(sample, sample) * dx)
                if ii + 1 in iterations:
                    expectation[size].append(sumL2/(ii + 1))
                if ii%(N//10) == 0 and comm.rank == 0:
                    print(100*(ii/N), '%', end=' ', flush=True)

            expectation[size] = np.array(expectation[size])
            if comm.rank == 0:
                print('100 %')
            EL2 = sumL2/N

        if comm.rank == 0:
            npzname = f'{args.dim}D_P{args.deg}_single_cvg{args.samples}'
            if args.seed != 123:
                npzname += f'_seed{args.seed}'
            np.savez_compressed(npzname,
                                iterations=iterations,
                                points=points,
                                point_vals=point_vals,
                                expectation=expectation,
                                volumes=volumes)

    if comm.rank == 0:
        plot_data(args, iterations, points, point_vals, expectation, volumes)

def plot_data(args, iterations, points, point_vals, expectation, volumes):
    factor = 2 if args.chop else 1
    mesh_size = [k for k in expectation.keys()]
    res = np.array([expectation[s] for s in mesh_size])
    ones = np.ones(len(iterations))
    vols = np.array([volumes[s] for s in mesh_size])
    analytic = np.abs(res - args.variance*np.outer(vols, ones))
    approx = np.abs(res - np.outer(res[:,-1], ones))

    breakpoint()

    for ii, (res, txt) in enumerate(zip([analytic, approx], ['\sigma^2|G|', '|| u_f ||_{{L^2}}'])):
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(6, 6)
        ax.loglog(iterations, res.T)
        ax.legend(np.array(mesh_size)/factor, title='Mesh Size')
        ax.set_xlabel('Samples')
        ax.set_ylabel(f'$|\mathbb{{E}}[|| u ||_{{L^2}} - {txt}]|$')
        pngname = f'{args.dim}D_P{args.deg}_single_cvg{args.samples}_{ii}'
        if args.seed != 123:
            pngname += f'_seed{args.seed}'
        pngname += '.png'
        fig.savefig(pngname, bbox_inches='tight', dpi=300)

    w = ceil(sqrt(len(mesh_size)))
    h = ceil(len(mesh_size)/w)
    figb, axb = plt.subplots(w, h, sharex=True, sharey=True)
    figb.set_size_inches(4*w, 4*h)

    Npoints = point_vals[mesh_size[0]].shape[1]
    dist = np.zeros((Npoints, Npoints))
    for ii, p in enumerate(points):
        dist[ii, :] = np.linalg.norm(points - p, axis=1)
    dist = np.tril(dist).ravel()
    dist = dist[dist>0]
    ind = np.argsort(dist)
    rmin = np.min(dist)
    rmax = np.max(dist)
    rspace = np.linspace(rmin, rmax, 100)

    for size, ax in zip(mesh_size, axb.ravel()):
        covar = np.cov(point_vals[size].T)
        cvlist = np.tril(covar, -1).ravel()
        cvlist = cvlist[cvlist!=0]

        ax.plot(dist, cvlist, 'rx')
        ax.plot(rspace, true_covariance(rspace))
        ax.set_xlabel('r')
        ax.set_ylabel('C(r)')
        ax.set_title(str(size/factor))
        ax.legend(['Samples', 'Exact'])

    figb.suptitle(f'Field convergence in {args.dim}D')
    pngname = f'{args.dim}D_P{args.deg}_covariance{args.samples}'
    if args.seed != 123:
        pngname += f'_seed{args.seed}'
    pngname += '.png'
    figb.savefig(pngname, bbox_inches='tight', dpi=300)

parser = ArgumentParser()
parser.add_argument('--samples',
                    default=1000,
                    type=int,
                    help='Number of samples to draw')
parser.add_argument('--seed',
                    default=123,
                    type=int,
                    help='Random number seed')
parser.add_argument('--dim',
                    default=2,
                    type=int,
                    choices=[2, 3],
                    help='Dimension')
parser.add_argument('--maxmesh',
                    default=100,
                    type=int,
                    help='Biggest mesh')
parser.add_argument('--deg',
                    default=1,
                    type=int,
                    help='Finite element degree (>=1)')
parser.add_argument('--smoothness',
                    default=1,
                    type=float,
                    help='Field smoothness')
parser.add_argument('--data',
                    default=None,
                    type=str,
                    help='filename.npz from a previous run')
parser.add_argument('--chop',
                    default=False,
                    action='store_true',
                    help='Remove boundary')


if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    _ = _main(args)
