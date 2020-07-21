import functools
import loopy

from firedrake import *
from firedrake.assemble import get_matrix, matrix_arg
from firedrake import tsfc_interface

from numpy.linalg import cholesky
import matplotlib.pyplot as plt

mesh = UnitSquareMesh(4, 4)
V = FunctionSpace(mesh, 'CG', 2)

pcg = PCG64(seed=100)
rg = RandomGenerator(pcg)
Wnoise = rg.normal(V, 0.0, 1.0)
HWnoise = Function(V)

u = TrialFunction(V)
v = TestFunction(V)

bcs = DirichletBC(V, 0, (1,2,3,4))

# Create mass expression, assemble and extract kernel
mass = inner(u,v)*dx
mass_ker, *stuff = tsfc_interface.compile_form(mass, "mass", coffee=False)

element_kernel = loopy.generate_code_v2(mass_ker.kinfo.kernel.code).device_code()
element_kernel = element_kernel.replace("void " + mass_ker.kinfo.kernel.name,
                                        "static void " + mass_ker.kinfo.kernel.name)

# Add custom code for doing Cholesky decomp
blocksize = mass_ker.kinfo.kernel.code.args[0].shape[0]

cholesky_code = f"""
extern void dpotrf_(char *UPLO,
                   int *N,
                   double *A,
                   int *LDA,
                   int *INFO);

{element_kernel}

void cholesky(double *__restrict__ H, double const *__restrict__ coords)
{{
    char uplo[1];
    int32_t N = {blocksize}, LDA = {blocksize}, INFO = 0;
    uplo[0] = 'u';

    {mass_ker.kinfo.kernel.name}(H, coords);
    dpotrf_(uplo, &N, H, &LDA, &INFO);

    for (int32_t i = 0; i < N; i++)
        for (int32_t j = 0; j < N; j++)
            if (j>i)
                H[i*N + j] = 0.0;

}}
"""

cholesky_kernel = op2.Kernel(cholesky_code, "cholesky", ldargs=["-llapack"])

# Construct an empty matrix like the mass matrix to store decomp
tensor, *_ = get_matrix(mass, bcs=bcs, mat_type='aij', sub_mat_type='aij')

# Construct arguments for par loop
def get_map(x):
    return x.cell_node_map()
i, j = mass_ker.indices

bcs = tuple(bc.extract_form('F') for bc in bcs)
create_op2arg = functools.partial(matrix_arg,
                                  #all_bcs=tuple(chain(*bcs)),
                                  matrix=tensor,
                                  Vrow=u.function_space(),
                                  Vcol=v.function_space())

tensor_arg = create_op2arg(op2.INC, get_map, i, j)

domain_number = mass_ker.kinfo.domain_number
domains = mass.ufl_domains()
m = domains[domain_number]
coords = m.coordinates

op2.par_loop(cholesky_kernel,
             mesh.cell_set,
             tensor_arg,
             coords.dat(op2.READ, get_map(coords)))


# Assemble original mass matrix and computed Cholesky factor
M = assemble(mass, mat_type='aij')
tensor.M.assemble()

# Plot comparing to numpy Cholesky factor
fig, ax = plt.subplots(1, 3)
ax[0].matshow(M.petscmat[:,:])
ax[0].set_title('Mass matrix')
ax[1].matshow(tensor.petscmat[:,:])
ax[1].set_title('Attempt')
ax[2].matshow(cholesky(M.petscmat[:,:]))
ax[2].set_title('True Cholesky')
plt.show()

# ~ with HWnoise.dat.vec_wo as y:
    # ~ with Wnoise.dat.vec_ro as x:
        # ~ H.mult(x, y)
