from argparse import ArgumentParser
from firedrake import *
from math import ceil, floor, sqrt

import numpy as np
import matplotlib.pyplot as plt

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
        volume = data['volume'][()]
        coarse_data = data['coarse_data'][()]
        fine_data = data['fine_data'][()]
    else:
        pcg = PCG64(seed=args.seed)
        rng = RandomGenerator(pcg)

        CHOP = args.chop
        dim = args.dim
        levels = args.levels
        if dim == 2:
            if CHOP:
                mesh = SquareMesh(args.baseN, args.baseN, 2)
            else:
                mesh = UnitSquareMesh(args.baseN, args.baseN)
        elif dim == 3:
            if CHOP:
                mesh = CubeMesh(args.baseN, args.baseN, args.baseN, 2)
            else:
                mesh = UnitCubeMesh(args.baseN, args.baseN, args.baseN)
        mh = MeshHierarchy(mesh, levels - 1)
        if CHOP:
            for m in mh:
                m.coordinates.dat.data[:, :] -= 0.5
        deg = args.deg
        smoothness = args.smoothness
        N = args.samples
        VAR = 1

        volume = {}
        fine_data = {}
        coarse_data = {}

        for l in range(levels - 1):
            fine_mesh = mh[l + 1]
            V_f = FunctionSpace(fine_mesh, 'CG', deg)
            GRF = GaussianRF(V_f, mu=0, sigma=VAR, correlation_length=0.2, smoothness=smoothness, rng=rng,
                            solver_parameters={'ksp_type': 'cg', 'pc_type': 'gamg', 'pc_gamg_threshold': -1})

            coarse_mesh = mh[l]
            V_c = FunctionSpace(coarse_mesh, 'CG', deg)
            sample_c = Function(V_c)

            fine_L2 = np.zeros(N)
            coarse_L2 = np.zeros(N)

            sample = indicator_f(Constant(1.0), fine_mesh)
            volume[l] = assemble(dot(sample, sample) * dx(domain=fine_mesh))

            if COMM_WORLD.rank == 0:
                print('Level:', l)
            for ii in range(N):
                sample_f = GRF.sample()
                inject(sample_f, sample_c)

                s_f = indicator_f(sample_f)
                s_c = indicator_f(sample_c)

                fine_L2[ii] = assemble(dot(s_f, s_f) * dx)
                coarse_L2[ii] = assemble(dot(s_c, s_c) * dx)

                if ii%(N//10) == 0 and COMM_WORLD.rank == 0:
                    print(100*(ii/N), '%', end=' ', flush=True)
            if COMM_WORLD.rank == 0:
                print('100 %')

            fine_data[l] = fine_L2
            coarse_data[l] = coarse_L2

        if comm.rank == 0:
            npzname = f'{args.dim}D_{args.baseN}_{args.levels}_P{args.deg}_ml_cvg{args.samples}'
            if args.seed != 123:
                npzname += f'_seed{args.seed}'
            np.savez_compressed(npzname,
                                fine_data=fine_data,
                                coarse_data=coarse_data,
                                volume=volume)

    if comm.rank == 0:
        # ~ results = [[0.12346840206238908, 0.0003043586021762483], [0.08947201607111559, 3.3405144111472866e-05], [0.04621505923167558, 5.054948927653155e-06], [0.01903046224171108, 3.396593296162226e-07]]
        plot_data(args, volume, fine_data, coarse_data)

def var_exp(a, axis=1, dtype=None, out=None, ddof=0):
    assert len(a.shape) == 2
    n = a.shape[axis]
    expn = ceil(sqrt(n))
    varn = floor(n/expn)

    if expn*varn != n:
        print(f'Warning: Throwing {n - expn*varn} samples away!')

    stack = a[:, :expn*varn].reshape(-1, varn, expn)
    return np.var(np.mean(stack, axis=2), axis=1)

def exp_var(a, axis=1, dtype=None, out=None, ddof=0):
    assert len(a.shape) == 2
    n = a.shape[axis]
    varn = ceil(sqrt(n))
    expn = floor(n/varn)

    if expn*varn != n:
        print(f'Warning: Throwing {n - expn*varn} samples away!')

    stack = a[:, :expn*varn].reshape(-1, expn, varn)
    return np.mean(np.var(stack, axis=2), axis=1)


def plot_data(args, volume, fine_data, coarse_data):
    levels = len(fine_data.keys())
    vol = np.array([v for v in volume.values()])
    fine = np.array([v for v in fine_data.values()])
    coarse = np.array([v for v in coarse_data.values()])
    ones = np.ones_like(fine[0, :])
    true_exp = np.abs(np.mean(args.variance*np.outer(vol, ones) - coarse, axis=1))
    num_exp = np.abs(np.mean(fine - coarse, axis=1))
    variance = np.var(fine - coarse, axis=1)

    # Consistency check
    print('Consistency check:')
    a = np.mean(fine - coarse, axis=1)
    b = np.mean(fine, axis=1)
    c = np.mean(coarse, axis=1)
    abc = a - b + c
    print('a - b + c = ', ' & '.join(f'{s:6.4g}' for s in abc))
    Va = var_exp(fine - coarse)
    Vb = var_exp(fine)
    Vc = var_exp(coarse)
    tabc = np.abs(abc)/(3*(np.sqrt(Va) + np.sqrt(Vb) + np.sqrt(Vc)))
    print('T(a, b, c) = ', ' & '.join(f'{s:6.4g}' for s in tabc))

    # Kurtosis
    k = exp_var((fine - coarse)**4)/(exp_var((fine - coarse)**2)**2)
    print('Kurtosis: ', ' & '.join(f'{s:6.4g}' for s in k))

    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(10, 6)
    p = (4 - args.dim)*args.deg
    if args.dim == 2:
        q = 2*p
    else:
        q = 2*(args.deg + 1)
    domain_length = 2 if args.chop else 1
    for res, ax, power in zip([true_exp, num_exp, variance], axes.ravel(), [p, p, q]):
        xs = [(domain_length/args.baseN)*2**(-l) for l in range(levels)]
        ax.loglog(xs, res.T)
        #power = p*(4 - args.dim)
        ax.loglog(xs, [2*(x**power)/(xs[-1]**power)*res[-1] for x in xs], 'k--')

        pos = (xs[-2], 2*(xs[-2]**power)/(xs[-1]**power)*res[-1])
        anchor = 'left'
        offset = (3, 2)
        ax.annotate(
            f'$h^{{{power}}}$',
            xy=pos, xycoords='data',
            xytext=offset, textcoords='offset points',
            fontsize='small', color='k', ha=anchor
        )
        ax.set_xscale("log", base=2)
        ax.set_xlabel('$h_l$')
        ax.invert_xaxis()


    axes[0].set_ylabel('$|\mathbb{E}[\sigma^2|G| - || u_{l-1} ||_{L^2}]|$')
    axes[1].set_ylabel('$|\mathbb{E}[|| u_l ||_{L^2} - || u_{l-1} ||_{L^2}]|$')
    axes[2].set_ylabel('$|\mathbb{V}[|| u_l ||_{L^2} - || u_{l-1} ||_{L^2}]|$')
    fig.suptitle(f'{args.dim}D multilevel field convergence with P{args.deg} elements\n Using {args.samples} samples per level')
    fig.subplots_adjust(wspace=0.4)
    pngname = f'{args.dim}D_{args.baseN}_{args.levels}_P{args.deg}_ml_cvg{args.samples}'
    if args.seed != 123:
        pngname += f'_seed{args.seed}'
    pngname += '.png'
    fig.savefig(pngname, dpi=300)


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
parser.add_argument('--baseN',
                    default=16,
                    type=int,
                    help='Smallest mesh size')
parser.add_argument('--levels',
                    default=7,
                    type=int,
                    help='Number of mesh refinements')
parser.add_argument('--deg',
                    default=1,
                    type=int,
                    help='Smallest degree (>=1)')
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
    if unknown:
        print('Unknown command line arguments:', unknown)
    _ = _main(args)
