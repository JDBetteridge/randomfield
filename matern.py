from firedrake import *

from math import ceil
from scipy.special import gamma
from firedrake.petsc import PETSc, get_petsc_variables
from firedrake.assemble import get_vector, vector_arg

import ufl
import numpy as np

_default_pcg = PCG64()

def matern(V, mean=0, variance=1, smoothness=1, correlation_length=1, rng=None, lognormal=False):
    '''
    Paper: Croci et al.
    https://arxiv.org/abs/1803.04857v2
    '''
    # Check for sensible arguments
    assert variance > 0
    assert smoothness > 0
    assert correlation_length > 0

    # Log normal rescaling
    if lognormal:
        mu = np.log(mean**2/np.sqrt(variance + mean**2))
        sigma = np.sqrt(np.log(1 + (variance/(mean**2))))
    else:
        mu = mean
        sigma = np.sqrt(variance)

    # Set symbols to match
    d = V.mesh().topological_dimension()
    nu = smoothness
    lambd = correlation_length
    k = ceil((nu + d/2)/2)

    # Calculate additional parameters
    kappa = np.sqrt(8*nu)/lambd
    sigma_hat2 = gamma(nu)*nu**(d/2)
    sigma_hat2 /= gamma(nu + d/2)
    sigma_hat2 *= (2/np.pi)**(d/2)
    sigma_hat2 *= lambd**(-d)
    eta = sigma/np.sqrt(sigma_hat2)

    # Print out parameters
    # ~ print('PARAMETERS:')
    # ~ print('Mean:', mu, 'Variance:', sigma)
    # ~ print('Smoothness:', smoothness, 'Correlation length', correlation_length)
    # ~ print('dimension:', d, 'iterations:', k)
    # ~ print('kappa:', kappa, 'sigma hat squared:', sigma_hat2, 'eta:', eta)
    # ~ print('kappa**-2:', 1/(kappa**2))

    # If no random number generator provided make a new one
    if rng is None:
        pcg = _default_pcg
        rng = RandomGenerator(pcg)

    # Setup modified Helmholtz problem
    u = TrialFunction(V)
    v = TestFunction(V)
    wnoise = white_noise(V, rng)
    a = (inner(u, v) + Constant(1/(kappa**2))*inner(grad(u), grad(v)))*dx
    l = Constant(eta)*inner(wnoise, v)*dx

    # Solve problem once
    u_h = Function(V)
    solver_param = {'ksp_type': 'cg', 'pc_type': 'gamg'}
    base_problem = LinearVariationalProblem(a, l, u_h)
    base_solver = LinearVariationalSolver(base_problem,
                                          solver_parameters=solver_param)
    base_solver.solve()

    # Iterate until required smoothness achieved
    if k>1:
        u_j = Function(V)
        u_j.assign(u_h)
        l_j = inner(u_j, v)*dx
        problem = LinearVariationalProblem(a, l_j, u_h)
        solver = LinearVariationalSolver(problem,
                                         solver_parameters=solver_param)
        for _ in range(k - 1):
            solver.solve()
            u_j.assign(u_h)

    if lognormal:
        u_h.dat.data[:] = np.exp(u_h.dat.data + mu)
    else:
        u_h.dat.data[:] = u_h.dat.data + mu
    return u_h

def white_noise(V, rng=None):
    '''
    '''
    # If no random number generator provided make a new one
    if rng is None:
        pcg = _default_pcg
        rng = RandomGenerator(pcg)

    # Create broken space for independent samples
    mesh = V.mesh()
    broken_elements = ufl.MixedElement([ufl.BrokenElement(Vi.ufl_element()) for Vi in V])
    Vbrok = FunctionSpace(mesh, broken_elements)
    iid_normal = rng.normal(Vbrok, 0.0, 1.0)
    wnoise = Function(V)

    # We also need cell volumes for correction
    DG0 = FunctionSpace(mesh, 'DG', 0)
    vol = Function(DG0)
    vol.interpolate(CellVolume(mesh))

    # Create mass expression, assemble and extract kernel
    u = TrialFunction(V)
    v = TestFunction(V)
    mass = inner(u,v)*dx
    mass_ker, *stuff = tsfc_interface.compile_form(mass, "mass", coffee=False)
    mass_code = loopy.generate_code_v2(mass_ker.kinfo.kernel.code).device_code()
    mass_code = mass_code.replace("void " + mass_ker.kinfo.kernel.name,
                                  "static void " + mass_ker.kinfo.kernel.name)

    # Add custom code for doing "Cholesky" decomp and applying to broken vector
    blocksize = mass_ker.kinfo.kernel.code.args[0].shape[0]

    cholesky_code = f"""
extern void dpotrf_(char *UPLO,
                    int *N,
                    double *A,
                    int *LDA,
                    int *INFO);

extern void dgemv_(char *TRANS,
                   int *M,
                   int *N,
                   double *ALPHA,
                   double *A,
                   int *LDA,
                   double *X,
                   int *INCX,
                   double *BETA,
                   double *Y,
                   int *INCY);

{mass_code}

void apply_cholesky(double *__restrict__ z,
                    double *__restrict__ b,
                    double const *__restrict__ coords,
                    double const *__restrict__ volume)
{{
    char uplo[1];
    int32_t N = {blocksize}, LDA = {blocksize}, INFO = 0;
    int32_t i=0, j=0;
    uplo[0] = 'u';
    double H[{blocksize}*{blocksize}] = {{{{ 0.0 }}}};

    char trans[1];
    int32_t stride = 1;
    //double one = 1.0;
    double scale = 1.0/volume[0];
    double zero = 0.0;

    {mass_ker.kinfo.kernel.name}(H, coords);

    uplo[0] = 'u';
    dpotrf_(uplo, &N, H, &LDA, &INFO);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (j>i)
                H[i*N + j] = 0.0;

    trans[0] = 'T';
    dgemv_(trans, &N, &N, &scale, H, &LDA, z, &stride, &zero, b, &stride);
}}
"""
    # Get the BLAS and LAPACK compiler parameters to compile the kernel
    if COMM_WORLD.rank == 0:
        petsc_variables = get_petsc_variables()
        BLASLAPACK_LIB = petsc_variables.get("BLASLAPACK_LIB", "")
        BLASLAPACK_LIB = COMM_WORLD.bcast(BLASLAPACK_LIB, root=0)
        BLASLAPACK_INCLUDE = petsc_variables.get("BLASLAPACK_INCLUDE", "")
        BLASLAPACK_INCLUDE = COMM_WORLD.bcast(BLASLAPACK_INCLUDE, root=0)
    else:
        BLASLAPACK_LIB = COMM_WORLD.bcast(None, root=0)
        BLASLAPACK_INCLUDE = COMM_WORLD.bcast(None, root=0)

    cholesky_kernel = op2.Kernel(cholesky_code,
                                 "apply_cholesky",
                                 include_dirs=BLASLAPACK_INCLUDE.split(),
                                 ldargs=BLASLAPACK_LIB.split())

    # Construct arguments for par loop
    def get_map(x):
        return x.cell_node_map()
    i, _ = mass_ker.indices

    z_arg = vector_arg(op2.READ, get_map, i, function=iid_normal, V=Vbrok)
    b_arg = vector_arg(op2.INC, get_map, i, function=wnoise, V=V)
    coords = mesh.coordinates
    volumes = vector_arg(op2.READ, get_map, i, function=vol, V=DG0)

    op2.par_loop(cholesky_kernel,
                 mesh.cell_set,
                 z_arg,
                 b_arg,
                 coords.dat(op2.READ, get_map(coords)),
                 volumes)

    return wnoise

