from argparse import ArgumentParser
from firedrake import *
from math import sqrt, ceil
from scipy.special import gamma
from scipy.special import kv as bessel2
from time import time

import numpy as np
import matplotlib.pyplot as plt

def true_covariance(r, sigma=1, nu=1, lambd=0.2):
    ''' Expression for evaluating the true covariance of a Matern field
    '''
    kappa=np.sqrt(8*nu)/lambd
    cov = sigma**2
    cov /= (2**(nu - 1))*gamma(nu)
    cov *= ((kappa*r)**nu)
    cov *= bessel2(nu, kappa*r)
    return cov

def indicator_f(f, mesh=None):
    ''' A symbolic indicator function on the unit square/cube
    '''
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

    if args.data:
        # If data is passed in populate variables and just plot
        data = np.load(args.data, allow_pickle=True)
        points = data['points'][()]
        point_vals = data['point_vals'][()]
        L2 = data['L2'][()]
        volumes = data['volumes'][()]
    else:
        pcg = PCG64(seed=args.seed)
        rng = RandomGenerator(pcg)

        dim = args.dim
        mesh_size = [2**(r + 1) for r in range(4, 15)]
        mesh_size = [m for m in mesh_size if m <= args.maxmesh]
        deg = args.deg
        N = args.samples
        smoothness = args.smoothness
        Npoints = 20
        if comm.rank == 0:
            points = rng.uniform(size=(Npoints, dim))
        else:
            points = np.zeros((Npoints, dim))
        comm.Bcast(points, root=0)
        volumes = {}
        L2 = {}
        point_vals = {}

        solver_param = {
            'ksp_type' : 'cg',
            'ksp_atol' : 1.0e-10,
            'ksp_rtol' : 1.0e-12,
            'ksp_norm_type' : 'unpreconditioned',
            'ksp_diagonal_scale' : True,
            'ksp_diagonal_scale_fix' : True,
            'ksp_reuse_preconditioner' : True,
            'ksp_max_it' : 1000,
            'pc_factor_mat_solver_type' : 'mumps',
            #'ksp_converged_reason' : None,
            #'ksp_monitor_true_residual' : None,
            #'ksp_view' : None,
            'pc_type' : 'hypre',
            'pc_hypre_boomeramg_strong_threshold' : [0.25, 0.25, 0.6][args.dim - 1],
            'pc_hypre_type' : 'boomeramg',
            }

        for size in mesh_size:
            if dim == 2:
                if args.chop:
                    mesh = SquareMesh(size, size, 2)
                    mesh.coordinates.dat.data[:, :] -= 0.5
                else:
                    mesh = UnitSquareMesh(size, size)
            elif dim == 3:
                if args.chop:
                    mesh = CubeMesh(size, size, size, 2)
                    mesh.coordinates.dat.data[:, :] -= 0.5
                else:
                    mesh = UnitCubeMesh(size, size, size)
            V = FunctionSpace(mesh, 'CG', deg)
            GRF = GaussianRF(
                V,
                mu=0,
                sigma=args.variance,
                smoothness=smoothness,
                correlation_length=0.2,
                rng=rng,
                solver_parameters=solver_param
                )

            # Vertex only mesh, to replace at()
            # ~ vom = VertexOnlyMesh(mesh, points)
            # ~ W = FunctionSpace(vom, 'DG', 0)
            # ~ interpolator = Interpolator(TestFunction(V), W)

            L2[size] = np.zeros(N)
            point_vals[size] = np.zeros((N, Npoints))
            sample = indicator_f(Constant(1.0), mesh=mesh)
            volumes[size] = assemble(dot(sample, sample) * dx(domain=mesh))

            if comm.rank == 0:
                print('Mesh size:', size)
                runtime = time()
            for ii in range(N):
                sample = GRF.sample()
                point_vals[size][ii, :] = sample.at(points)
                # ~ w = interpolator.interpolate(sample)
                # ~ point_vals[size][ii, :] = w.dat.data_ro
                sample = indicator_f(sample)
                L2[size][ii] = assemble(dot(sample, sample) * dx)
                if ii%(N//10) == 0 and comm.rank == 0:
                    print(100*(ii/N), '%', end=' ', flush=True)

            if comm.rank == 0:
                print('100 %')
                runtime = time() - runtime
                print('Runtime : ', runtime, 's')

        if comm.rank == 0:
            npzname = f'{args.dim}D_P{args.deg}_single_cvg{args.samples}'
            if args.seed != 123:
                npzname += f'_seed{args.seed}'
            np.savez_compressed(npzname,
                                points=points,
                                point_vals=point_vals,
                                L2=L2,
                                volumes=volumes)

    if comm.rank == 0:
        plot_data(args, points, point_vals, L2, volumes)

def plot_data(args, points, point_vals, L2, volumes):
    factor = 2 if args.chop else 1
    mesh_size = [k for k in L2.keys()]
    L2 = np.array([L2[s] for s in mesh_size])
    iterations = np.arange(1, L2.shape[1]+1)
    ones = np.ones_like(L2[0])
    vols = np.array([volumes[s] for s in mesh_size])

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

    for ii, (res, txt) in enumerate(zip([analytic, approx], ['\sigma^2|G|', '|| u_f ||_{{L^2}}^2'])):
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(6, 6)
        # Data
        ax.plot(res.T)
        # Add confidence intervals
        ax.set_prop_cycle(None)
        ax.plot(+3*np.sqrt(np.outer(P_var, 1/iterations)).T, ls='--', lw=1)
        ax.set_prop_cycle(None)
        ax.plot(-3*np.sqrt(np.outer(P_var, 1/iterations)).T, ls='--', lw=1)
        # Labels and legends
        ax.legend(np.array(mesh_size)/factor, title='Mesh Size')
        ax.set_xlabel('Samples')
        ax.set_ylabel(f'$|\mathbb{{E}}[|| u ||_{{L^2}}^2 - {txt}]|$')
        pngname = f'{args.dim}D_P{args.deg}_single_cvg{args.samples}_{ii}'
        if args.seed != 123:
            pngname += f'_seed{args.seed}'
        pngname += '.png'
        fig.savefig(pngname, bbox_inches='tight', dpi=300)

    for ii, (res, txt) in enumerate(zip([analytic, approx], ['\sigma^2|G|', '|| u_f ||_{{L^2}}^2'])):
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(6, 6)
        ax.loglog(res[-1])
        ax.loglog(+3*np.sqrt(P_var[-1]/iterations), 'k--')
        ax.loglog(-3*np.sqrt(P_var[-1]/iterations), 'k--')
        ax.legend([np.array(mesh_size[-1])/factor], title='Mesh Size')
        ax.set_xlabel('Samples')
        ax.set_ylabel(f'$|\mathbb{{E}}[|| u ||_{{L^2}}^2 - {txt}]|$')
        pngname = f'finest_{args.dim}D_P{args.deg}_single_cvg{args.samples}_{ii}'
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
    rmin = np.min(dist)
    rmax = np.max(dist)
    rspace = np.linspace(rmin, rmax, 100)

    for size, ax in zip(mesh_size, axb.ravel()):
        covar = np.cov(point_vals[size].T)
        # ~ var = np.diag(covar)
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
parser.add_argument('--variance',
                    default=1,
                    type=float,
                    help='Variance (probably don\'t change this)')


if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    _ = _main(args)
