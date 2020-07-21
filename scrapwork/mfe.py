from firedrake import *

mesh = UnitSquareMesh(25, 25)
V = FunctionSpace(mesh, 'CG', 2)

pcg = PCG64(seed=100)
rg = RandomGenerator(pcg)
Wnoise = rg.normal(V, 0.0, 1.0)
HWnoise = Function(V)

u = TrialFunction(V)
v = TestFunction(V)
mass = inner(u,v)*dx
A = assemble(mass, mat_type='aij').M.handle
pc = PETSc.PC().create(COMM_WORLD)
pc.setOperators(A)
pc.setType("cholesky")
pc.setFactorSolverType("mumps")
pc.setUp()
F = pc.getFactorMatrix()

with Wnoise.dat.vec_ro as x:
    with HWnoise.dat.vec_wo as y:
        F.mult(x, y)
