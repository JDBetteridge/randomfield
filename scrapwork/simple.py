from firedrake import *

from numpy.linalg import cholesky
from firedrake.petsc import PETSc

import ufl

import matplotlib.pyplot as plt

## Extrct the PETSc matrix
def get_PETSC_matrix(form, **kwargs):
    operator = assemble(form, **kwargs).M.handle
    return operator

## Convert to numpy matrix
def get_np_matrix(form, mat=None, **kwargs):
    if mat is None:
        L = get_PETSC_matrix(form, **kwargs)
    else:
        L = mat

    size = L.getSize()
    Lij = PETSc.Mat()
    L.convert('aij', Lij)

    Lnp = np.array(Lij.getValues(range(size[0]), range(size[1])))
    return Lnp

fig, ax = plt.subplots(1, 3)

mesh = UnitSquareMesh(25, 25)

# ~ cgelt = FiniteElement('CG', 'triangle', 2)
# ~ broken_element = ufl.BrokenElement(cgelt)
# ~ V = FunctionSpace(mesh, broken_element)
V = FunctionSpace(mesh, 'CG', 2)

pcg = PCG64(seed=100)
rg = RandomGenerator(pcg)
Wnoise = rg.normal(V, 0.0, 1.0)
HWnoise = Function(V)
tripcolor(Wnoise, axes=ax[0])

u = TrialFunction(V)
v = TestFunction(V)

bcs = DirichletBC(V, 0, (1,2,3,4))

mfig, mmax = plt.subplots(1, 2)
mass = inner(u,v)*dx


# ~ breakpoint()
# ~ M = assemble(mass, bcs=bcs, mat_type='aij').M.handle
# ~ Mnp = get_np_matrix(0, mat=M)
# ~ pc = PETSc.PC().create(COMM_WORLD)
# ~ pc.setOperators(M)
# ~ pc.setType("cholesky")
# ~ pc.setFactorSolverType("petsc")
# ~ pc.setUp()
# ~ H = pc.getFactorMatrix()


#breakpoint()
#Hnp = get_np_matrix(0, mat=H)

with HWnoise.dat.vec_wo as y:
    with Wnoise.dat.vec_ro as x:
        H.mult(x, y)

# ~ M = get_np_matrix(mass)
# ~ H = cholesky(M)
mmax[0].matshow(Mnp)
mmax[1].matshow(Hnp)
# ~ HWnoise.dat.data[:] = H@Wnoise.dat.data
#print(np.sum(HWnoise.dat.data - Wnoise.dat.data))
tripcolor(HWnoise, axes=ax[1])
#M = assemble(mass, bcs=bcs)
#breakpoint()

a = (inner(u, v) + inner(grad(u), grad(v)))*dx
l = inner(HWnoise, v)*dx

u_h = Function(V)
solve(a == l, u_h, bcs=bcs, solver_parameters={'ksp_type': 'cg', 'pc_type': 'gamg'})



tripcolor(u_h, axes=ax[2])
plt.show()
