import functools
import loopy
import ufl

from firedrake import *
from firedrake.assemble import get_matrix, matrix_arg, get_vector, vector_arg
from firedrake import tsfc_interface

import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt

# Set up RNG
pcg = PCG64(seed=100)
rg = RandomGenerator(pcg)

samples = 2000
npmat = np.zeros((6, 6))

for ii in range(samples):
    mesh = UnitSquareMesh(1, 1)
    mesh = UnitTriangleMesh()
    V = FunctionSpace(mesh, 'CG', 2)

    Wnoise = rg.normal(V, 0.0, 1.0)
    u = TrialFunction(V)
    v = TestFunction(V)

    # ~ ax[ii].matshow(Wnoise.dat.data.reshape(Wnoise.dat.data.size, 1))
    print(Wnoise.dat.data)
    #bcs = DirichletBC(V, 0, (1,2,3,4))

    # Create mass expression, assemble and extract kernel
    mass = inner(u,v)*dx
    mass_ker, *stuff = tsfc_interface.compile_form(mass, "mass", coffee=False)

    mat_kernel = loopy.generate_code_v2(mass_ker.kinfo.kernel.code).device_code()
    mat_kernel = mat_kernel.replace("void " + mass_ker.kinfo.kernel.name,
                                    "static void " + mass_ker.kinfo.kernel.name)

    # Create rhs expression, assemble and extract kernel
    rhs = inner(Wnoise, v)*dx
    rhs_ker, *stuff = tsfc_interface.compile_form(rhs, "rhs", coffee=False)

    vec_kernel = loopy.generate_code_v2(rhs_ker.kinfo.kernel.code).device_code()
    vec_kernel = vec_kernel.replace("void " + rhs_ker.kinfo.kernel.name,
                                    "static void " + rhs_ker.kinfo.kernel.name)

    # Add custom code for doing Cholesky decomp
    blocksize = rhs_ker.kinfo.kernel.code.args[0].shape[0]
    coordlen = blocksize = rhs_ker.kinfo.kernel.code.args[1].shape[0]

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

{vec_kernel}
{mat_kernel}

void apply_cholesky(double *__restrict__ z, double const *__restrict__ coords)
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
    double tmp[{blocksize}] = {{{{ 0.0 }}}};
    double w_0[{blocksize}] = {{{{ 1.0 }}}};

    {mass_ker.kinfo.kernel.name}(H, coords);

    /* for(i=0; i<N; i++){{
        for(j=0; j<N; j++){{
            printf("%g\\t", H[N*i+j]);
        }}
        printf("\\n");
    }}
    printf("\\n"); */

    uplo[0] = 'u';
    dpotrf_(uplo, &N, H, &LDA, &INFO);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (j>i)
                H[i*N + j] = 0.0;


    /* for(i=0; i<N; i++){{
        for(j=0; j<N; j++){{
            printf("%g\\t", H[N*i+j]);
        }}
        printf("\\n");
    }}
    printf("\\n"); */

    /*{rhs_ker.kinfo.kernel.name}(x, coords, w_0);*/

    for(i=0; i<N; i++){{
        printf("%g\\t %g\\n", z[i], tmp[i]);
    }}
    printf("\\n");

    trans[0] = 'T';
    dgemv_(trans, &N, &N, &one, H, &LDA, z, &stride, &zero, tmp, &stride);

    for(i=0; i<N; i++){{
        printf("%g\\t %g\\n", z[i], tmp[i]);
        z[i] = tmp[i];
    }}
    printf("\\n");
}}
"""

    cholesky_kernel = op2.Kernel(cholesky_code, "apply_cholesky", ldargs=["-llapack", "-lblas"])

    # Construct an empty matrix like the mass matrix to store decomp

    mat, *_ = get_matrix(mass, mat_type='aij', sub_mat_type='aij') #bcs?

    test, = rhs.arguments()
    vec, *_ = get_vector(test)


    # Construct arguments for par loop
    def get_map(x):
        return x.cell_node_map()
    i, j = mass_ker.indices
    i, = rhs_ker.indices

    #bcs = tuple(bc.extract_form('F') for bc in bcs)
    mat_create_op2arg = functools.partial(matrix_arg,
                                      #all_bcs=tuple(chain(*bcs)),
                                      matrix=mat,
                                      Vrow=v.function_space(),
                                      Vcol=u.function_space())
    mat_arg = mat_create_op2arg(op2.INC, get_map, i, j)

    vec_create_op2arg = functools.partial(vector_arg, function=vec, V=test.function_space())
    vec_arg = vec_create_op2arg(op2.RW, get_map, i)
    vec2_arg = vector_arg(op2.RW, get_map, i, function=Wnoise, V=test.function_space())

    domain_number = mass_ker.kinfo.domain_number
    domains = mass.ufl_domains()
    m = domains[domain_number]
    coords = m.coordinates

    op2.par_loop(cholesky_kernel,
                 mesh.cell_set,
                 vec2_arg,
                 coords.dat(op2.READ, get_map(coords)))
    npmat += np.outer(Wnoise.dat.data, Wnoise.dat.data)
    print(Wnoise.dat.data)



# Assemble original mass matrix and computed Cholesky factor
M = assemble(mass, mat_type='aij')
mat.M.assemble()

z = assemble(rhs)

fig, ax = plt.subplots(1, 3)
Mm = np.ma.masked_values(M.petscmat[:,:], 0, atol=1e-15)
ax[0].matshow(Mm)
npm = np.ma.masked_values(npmat/samples, 0, atol=1e-15)
ax[1].matshow(npm)
diffm = np.ma.masked_values(M.petscmat[:,:] - npmat/samples, 0, atol=1e-15)
ax[2].matshow(diffm)
plt.show()



# Plot comparing to numpy Cholesky factor
# ~ fig, ax = plt.subplots(2, 4)
# ~ npchol = cholesky(M.petscmat[:,:])

# ~ for ii, A in enumerate([tensor.petscmat[:,:], npchol]):
    # ~ Mm = np.ma.masked_values(M.petscmat[:,:], 0, atol=1e-15)
    # ~ ax[ii,0].matshow(Mm)
    # ~ Am = np.ma.masked_values(A, 0, atol=1e-15)
    # ~ ax[ii,1].matshow(Am)
    # ~ AATm = np.ma.masked_values(A@A.T, 0, atol=1e-15)
    # ~ ax[ii,2].matshow(AATm)
    # ~ Dm = np.ma.masked_values(M.petscmat[:,:] - A@A.T, 0, atol=1e-15)
    # ~ ax[ii,3].matshow(Dm)

# ~ ax[0,0].set_title('Mass matrix $M$')
# ~ ax[0,1].set_title('Decomp $A$')
# ~ ax[0,2].set_title('$AA^T$')
# ~ ax[0,3].set_title('$M - AA^T$')

# ~ ax[0,0].set_ylabel('My Kernel')
# ~ ax[1,0].set_ylabel('numpy')
# ~ plt.show()

# ~ with HWnoise.dat.vec_wo as y:
    # ~ with Wnoise.dat.vec_ro as x:
        # ~ tensor.M.handle.mult(x, y)

#HWnoise.dat.data = npchol@HWnoise.dat.data
