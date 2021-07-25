from argparse import ArgumentParser
from firedrake import *
from time import time

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

    if args.data:
        # If data is passed in populate variables and just plot
        data = np.load(args.data, allow_pickle=True)
        volume = data['volume'][()]
        coarse_data = data['coarse_data'][()]
        fine_data = data['fine_data'][()]
    else:
        pcg = PCG64(seed=args.seed)
        rng = RandomGenerator(pcg)

        # Construct hierarchy of meshes, with padded boundary if chop specified
        dim = args.dim
        levels = args.levels
        if dim == 2:
            if args.chop:
                mesh = SquareMesh(args.baseN, args.baseN, 2)
            else:
                mesh = UnitSquareMesh(args.baseN, args.baseN)
        elif dim == 3:
            if args.chop:
                mesh = CubeMesh(args.baseN, args.baseN, args.baseN, 2)
            else:
                mesh = UnitCubeMesh(args.baseN, args.baseN, args.baseN)
        mh = MeshHierarchy(mesh, levels - 1)
        if args.chop:
            for m in mh:
                m.coordinates.dat.data[:, :] -= 0.5
        deg = args.deg
        smoothness = args.smoothness
        N = args.samples

        volume = {}
        fine_data = {}
        coarse_data = {}

        for l in range(levels - 1):
            # Set up random field on each level
            fine_mesh = mh[l + 1]
            V_f = FunctionSpace(fine_mesh, 'CG', deg)
            param = {
                'ksp_type': 'cg',
                'pc_type': 'gamg',
                'pc_gamg_threshold': -1
            }
            GRF = GaussianRF(
                V_f,
                mu=0,
                sigma=args.variance,
                correlation_length=0.2,
                smoothness=smoothness,
                rng=rng,
                solver_parameters=param
                )

            coarse_mesh = mh[l]
            V_c = FunctionSpace(coarse_mesh, 'CG', deg)
            sample_c = Function(V_c)

            fine_L2 = np.zeros(N)
            coarse_L2 = np.zeros(N)

            sample = indicator_f(Constant(1.0), fine_mesh)
            volume[l] = assemble(dot(sample, sample) * dx(domain=fine_mesh))

            if COMM_WORLD.rank == 0:
                print('Level:', l)
                runtime = time()
            for ii in range(N):
                # Draw a sample on the fine mesh and inject to coarse mesh
                sample_f = GRF.sample()
                inject(sample_f, sample_c)

                s_f = indicator_f(sample_f)
                s_c = indicator_f(sample_c)

                # Measure and record the L2-norm squared
                fine_L2[ii] = assemble(dot(s_f, s_f) * dx)
                coarse_L2[ii] = assemble(dot(s_c, s_c) * dx)

                if ii%(N//10) == 0 and COMM_WORLD.rank == 0:
                    print(100*(ii/N), '%', end=' ', flush=True)
            if COMM_WORLD.rank == 0:
                print('100 %')
                runtime = time() - runtime
                print('Runtime : ', runtime, 's')

            fine_data[l] = fine_L2
            coarse_data[l] = coarse_L2

        # Save the array of L2 norms as checkpointing
        if comm.rank == 0:
            npzname = f'{args.dim}D_{args.baseN}_{args.levels}_P{args.deg}_ml_cvg{args.samples}'
            if args.seed != 123:
                npzname += f'_seed{args.seed}'
            np.savez_compressed(npzname,
                                fine_data=fine_data,
                                coarse_data=coarse_data,
                                volume=volume)

    if comm.rank == 0:
        plot_data(args, volume, fine_data, coarse_data)


def plot_data(args, volume, fine_data, coarse_data):
    try:
        from scipy.stats import kurtosis
    except ImportError:
        # Gives same values as scipy for test values
        kurtosis = lambda x: np.mean((x - np.outer(np.mean(x, axis=1), np.ones_like(x[0])))**4, axis=1)/(np.var(x, axis=1)**2)
    levels = len(fine_data.keys())
    vol = np.array([v for v in volume.values()])
    fine = np.array([v for v in fine_data.values()])
    coarse = np.array([v for v in coarse_data.values()])

    # Statistics in MLMC are often calculated in terms of
    # Y_l = P_l      - P_{l-1}
    #     = P_l^f    - P_{l-1}^c
    #     = P_l^f(w) - P_{l-1}^c(w)
    # where l denotes the level and c or f denotes fine or coarse
    # w (in place of \omega) denotes the statistic is generated
    # from the same sample
    Y = fine - coarse
    Pl = fine

    ones = np.ones_like(fine[0, :])
    true_exp = np.abs(np.mean(args.variance*np.outer(vol, ones) - coarse, axis=1))
    Y_mean = np.mean(Y, axis=1)
    Y_var = np.var(Y, axis=1)

    # P_l statistics
    print('P_l statistics')
    P_mean = np.mean(Pl, axis=1)
    P_var = np.var(Pl, axis=1)
    print('Mean       : ', ' & '.join(f'{s:6.4g}' for s in P_mean))
    print('Variance   : ', ' & '.join(f'{s:6.4g}' for s in P_var))

    # Y_l statistics
    # Consistency check
    print('Y_l statistics')
    fine_exp = np.mean(fine, axis=1)
    a = Y_mean[1:]
    b = fine_exp[1:]
    c = fine_exp[:-1]
    abc = a - b + c
    print('a - b + c  = ', ' & '.join(f'{s:6.4g}' for s in abc))
    fine_var = np.var(fine, axis=1)
    Va = Y_var[1:]
    Vb = fine_var[1:]
    Vc = fine_var[:-1]
    N = Y.shape[1]
    tabc = np.sqrt(N)*np.abs(abc)/(3*(np.sqrt(Va) + np.sqrt(Vb) + np.sqrt(Vc)))
    print('T(a, b, c) = ', ' & '.join(f'{s:6.4g}' for s in tabc))

    # Kurtosis
    Y_kurt = kurtosis(fine - coarse, axis=1, fisher=False)
    print('Mean       : ', ' & '.join(f'{s:6.4g}' for s in Y_mean))
    print('Variance   : ', ' & '.join(f'{s:6.4g}' for s in Y_var))
    print('Kurtosis   : ', ' & '.join(f'{s:6.4g}' for s in Y_kurt))

    fig, axes = plt.subplots(2, 3)
    fig.set_size_inches(10, 10)

    # Convergence plots
    p = (4 - args.dim)*args.deg
    if args.dim == 2:
        q = 2*p
    else:
        q = 2*(args.deg + 1)
    domain_length = 2 if args.chop else 1
    xs = [(domain_length/args.baseN)*2**(-l) for l in range(levels)]
    for res, ax, power in zip([true_exp, np.abs(Y_mean), Y_var], axes[0].ravel(), [p, p, q]):
        # Data
        ax.loglog(xs, res.T)
        #power = p*(4 - args.dim)
        # Rate
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

    axes[0, 0].set_ylabel('$|\mathbb{E}[\sigma^2|G| - || u_{l-1} ||_{L^2}^2]|$')
    axes[0, 1].set_ylabel('$|\mathbb{E}[|| u_l ||_{L^2}^2 - || u_{l-1} ||_{L^2}^2]|$')
    axes[0, 2].set_ylabel('$|\mathbb{V}[|| u_l ||_{L^2}^2 - || u_{l-1} ||_{L^2}^2]|$')

    # Consistency and kurtosis
    axes[1, 0].plot(range(1, levels), tabc)
    axes[1, 0].set_xlabel('Level')
    axes[1, 0].set_ylabel('Consistency $T(a,b,c)$')

    axes[1, 1].plot(range(levels), Y_kurt)
    axes[1, 1].set_xlabel('Level')
    axes[1, 1].set_ylabel('Kurtosis')

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
parser.add_argument('--variance',
                    default=1,
                    type=float,
                    help='Variance (probably don\'t change this)')


if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Unknown command line arguments:', unknown)
    _ = _main(args)
