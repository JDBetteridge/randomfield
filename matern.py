from firedrake import *

from numpy.linalg import cholesky
from scipy.special import gamma
from firedrake.petsc import PETSc
from firedrake.assemble import get_vector, vector_arg

import ufl
import numpy as np

def matern(V, variance=1, smoothness=1, correlation_length=1, rng=None):
    '''
    '''
    d = V.mesh().topological_dimension()
    nu = smoothness
    sigma = np.sqrt(variance)
    lambd = correlation_length

    k = (nu + d/2)/2
    assert k==1
    kappa = np.sqrt(8)/lambd
    sigma_hat = gamma(nu)*nu**(d/2)
    sigma_hat /= gamma(nu + d/2)
    sigma_hat *= (2/np.pi)**(d/2)
    sigma_hat *= lambd**(-d)
    eta = sigma/sigma_hat

    if rng is None:
        pcg = PCG64(seed=100)
        rng = RandomGenerator(pcg)

    u = TrialFunction(V)
    v = TestFunction(V)
    wnoise = white_noise(V, rng)

    a = (inner(u, v) + Constant(1/(kappa**2))*inner(grad(u), grad(v)))*dx
    l = Constant(eta)*inner(wnoise, v)*dx

    u_h = Function(V)
    solve(a == l, u_h, solver_parameters={'ksp_type': 'cg', 'pc_type': 'gamg'})

    return u_h

def white_noise(V, rng):
    '''
    '''
    # Create broken space for independent samples
    mesh = V.mesh()
    broken_elements = ufl.MixedElement([ufl.BrokenElement(Vi.ufl_element()) for Vi in V])
    Vbrok = FunctionSpace(mesh, broken_elements)
    iid_normal = rng.normal(Vbrok, 0.0, 1.0)
    wnoise = Function(V)

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
                    double const *__restrict__ coords)
{{
    char uplo[1];
    int32_t N = {blocksize}, LDA = {blocksize}, INFO = 0;
    int32_t i=0, j=0;
    uplo[0] = 'u';
    double H[{blocksize}*{blocksize}] = {{{{ 0.0 }}}};

    char trans[1];
    int32_t stride = 1;
    double one = 1.0;
    double zero = 0.0;

    {mass_ker.kinfo.kernel.name}(H, coords);

    uplo[0] = 'u';
    dpotrf_(uplo, &N, H, &LDA, &INFO);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (j>i)
                H[i*N + j] = 0.0;

    trans[0] = 'T';
    dgemv_(trans, &N, &N, &one, H, &LDA, z, &stride, &zero, b, &stride);
}}
"""

    cholesky_kernel = op2.Kernel(cholesky_code, "apply_cholesky", ldargs=["-llapack", "-lblas"])

    # Construct arguments for par loop
    def get_map(x):
        return x.cell_node_map()
    i, _ = mass_ker.indices

    z_arg = vector_arg(op2.READ, get_map, i, function=iid_normal, V=Vbrok)
    b_arg = vector_arg(op2.INC, get_map, i, function=wnoise, V=V)

    # ~ domain_number = mass_ker.kinfo.domain_number
    # ~ domains = mass.ufl_domains()
    # ~ m = domains[domain_number]
    coords = mesh.coordinates

    op2.par_loop(cholesky_kernel,
                 mesh.cell_set,
                 z_arg,
                 b_arg,
                 coords.dat(op2.READ, get_map(coords)))

    return wnoise

