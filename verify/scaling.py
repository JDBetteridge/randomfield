from argparse import ArgumentParser
from firedrake import *
from pathlib import Path
from time import time

import csv
import numpy as np
import matplotlib.pyplot as plt

def _main(args):
    if args.data:
        plot_data(args)
    else:
        nsample = args.samples
        nx = args.baseN
        tsfactor = args.tsfactor

        corr_len = 0.2
        smoothness = args.smoothness

        # Start timing at mesh construction
        total_time = time()
        if args.dim == 2:
            mesh = UnitSquareMesh(nx, nx)
        elif args.dim == 3:
            mesh = UnitCubeMesh(nx, nx, nx)
        mh = MeshHierarchy(mesh, args.levels - 1)
        V = FunctionSpace(mh[-1], 'CG', args.deg)

        # Construct many samples of LogNormal random fields
        # Using mean and standard deviation parameters
        pcg = PCG64(seed=123)
        # Mat free CG + GMG
        # ~ solver_opts = {
            # ~ "ksp_type": "cg",
            # ~ "pc_type": "mg",
            # ~ "pc_mg_type": "full",
            # ~ "mg_levels_ksp_type": "chebyshev",
            # ~ "mg_levels_ksp_max_it": 2,
            # ~ "mg_levels_pc_type": "jacobi",
            # ~ "mg_coarse_pc_type": "python",
            # ~ "mg_coarse_pc_python_type": "firedrake.AssembledPC",
            # ~ "mg_coarse_assembled": {
                # ~ "mat_type": "aij",
                # ~ "pc_type": "telescope",
                # ~ "pc_telescope_reduction_factor": tsfactor,
                # ~ "pc_telescope_subcomm_type": "contiguous",
                # ~ "telescope_pc_type": "lu",
                # ~ "telescope_pc_factor_mat_solver_type": "mumps"
            # ~ }
        # ~ }
        # Mat free CG + GMG
        solver_opts = {
            'snes_view': None,
            'ksp_type': 'cg',
            'pc_type': 'gamg'
        }
        # Just MG
        # ~ solver_opts = {
            # ~ 'snes_view': None,
            # ~ 'ksp_type': 'preonly',
            # ~ 'pc_type': 'mg',
            # ~ 'pc_mg_log': None,
            # ~ 'pc_mg_type': 'full',
            # ~ 'mg_levels_ksp_type': 'chebyshev',
            # ~ 'mg_levels_ksp_max_it': 2,
            # ~ 'mg_levels_pc_type': 'jacobi',
            # ~ 'mg_coarse_pc_type': 'lu',
            # ~ 'mg_coarse_pc_factor_mat_solver_type': 'mumps'
        # ~ }
        # Just CG
        # ~ solver_opts = {
            # ~ 'snes_view': None,
            # ~ 'ksp_type': 'cg',
            # ~ 'pc_type': 'ilu'
        # ~ }
        # ~ solver_opts = {
            # ~ 'snes_view': None,
            # ~ 'ksp_type': 'preonly',
            # ~ 'pc_type': 'lu',
            # ~ 'pc_factor_mat_solver_type': 'mumps'
        # ~ }
        rng = RandomGenerator(pcg)
        LNRF = LogGaussianRF(V, mean=5, std_dev=1,
                             smoothness=smoothness,
                             correlation_length=corr_len,
                             solver_parameters=solver_opts,
                             V_aux=None)
        L2 = np.zeros(nsample)
        for ii in range(nsample):
            lognormal_rf = LNRF.sample(rng)
            L2[ii] = 0 #assemble(dot(lognormal_rf, lognormal_rf) * dx)

        total_time = time() - total_time

        if COMM_WORLD.rank == 0:
            write_data(args, LNRF, nsample, L2, total_time)

def write_data(args, randomfield, nsample, L2, total_time):
    print('mu =', randomfield.mu, 'sigma =', randomfield.sigma)
    print('field average =', np.average(randomfield.u_h.dat.data),
          'field variance = ', np.var(randomfield.u_h.dat.data),
          'field standard deviation = ', np.std(randomfield.u_h.dat.data)
    )

    print('L2 average =', np.average(L2),
          'L2 variance = ', np.var(L2),
          'L2 standard deviation = ', np.std(L2)
    )
    print('Num Samples:', nsample,
          'Total time:', total_time,
          'Average time', total_time/nsample)

    filename = args.output
    fields = [
        'nproc', 'nsample', 'baseN', 'levels', 'ndof', 'auxdof', 'dim',
        'mu', 'sigma', 'smoothness', 'correlation length',
        'L2 avg', 'L2 std', 'time'
    ]
    filepath = Path(filename).absolute()
    new = not filepath.exists()
    with open(filepath, 'a') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fields)
        if new:
            writer.writeheader()
        result = {
            'nproc': COMM_WORLD.size,
            'nsample': nsample,
            'baseN': args.baseN,
            'levels': args.levels,
            'ndof': randomfield.V.dim(),
            'auxdof': 0,
            'dim': args.dim,
            'mu': randomfield.mu,
            'sigma': randomfield.sigma,
            'smoothness': randomfield.nu,
            'correlation length': randomfield.lambd,
            'L2 avg': np.average(L2),
            'L2 std': np.std(L2),
            'time': total_time
        }
        writer.writerow(result)

def plot_data(args):
    from pandas import DataFrame, read_csv
    from pathlib import Path
    import plot_tools as stdplot

    def fn2case(filename):
        p = Path(filename)
        name = p.stem
        case = name[0:5].upper()
        case = case.replace('_', ' ')
        return case


    raw_data = DataFrame()
    col = 'case'
    for csv in args.data:
        csv_data = read_csv(csv)
        csv_data[col] = fn2case(csv)
        raw_data = raw_data.append(csv_data)

    data = raw_data.pivot(index='nproc', columns=col, values='time')
    dof_data = raw_data.pivot(index='nproc', columns=col, values='ndof')

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches((6, 6))

    stdplot.strong_scaling_plot(ax, data, select=data.columns,
                            legends='Field', dofs=None)
    for ii in range(len(data.columns)):
        stdplot.add_dofs(
            ax, data, dof_data, data.columns[ii],
            color='k', outline='w',
            reverse_last=True
        )

    ax.set_title(args.title)

    pngname = Path(args.output).stem
    pngname += '.png'
    fig.savefig(pngname, dpi=300)


parser = ArgumentParser()
parser.add_argument('--samples',
                    default=128,
                    type=int,
                    help='Number of samples to draw')
parser.add_argument('--dim',
                    default=3,
                    type=int,
                    choices=[2, 3],
                    help='Dimension')
parser.add_argument('--baseN',
                    default=32,
                    type=int,
                    help='Smallest mesh size')
parser.add_argument('--levels',
                    default=2,
                    type=int,
                    help='Number of mesh refinements')
parser.add_argument('--deg',
                    default=1,
                    type=int,
                    help='Smallest degree (>=1)')
parser.add_argument('--smoothness',
                    default=0.5,
                    type=float,
                    help='Field smoothness')
parser.add_argument('--output',
                    default='fieldscaling.csv',
                    type=str,
                    help='filename.csv from a scaling run')
parser.add_argument('--data',
                    default=None,
                    nargs='*',
                    help='filename.csv from previous runs')
parser.add_argument('--tsfactor',
                    default=1,
                    type=int,
                    help='Telescoping factor')
parser.add_argument('--title',
                    default='Scaling',
                    type=str,
                    help='title for plot')


if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Unknown command line arguments:', unknown)
    _ = _main(args)
