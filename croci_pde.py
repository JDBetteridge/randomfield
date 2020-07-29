from firedrake import *
from matern import matern

import matplotlib.pyplot as plt

mesh = UnitSquareMesh(50, 50)
V = FunctionSpace(mesh, 'CG', 2)

q = TrialFunction(V)
p = TestFunction(V)
q_h = Function(V)
q_hh = Function(V)

u = matern(V, mean=1, variance=0.2, correlation_length=0.1)
a = inner(exp(u)*grad(q), grad(p))*dx
l = inner(Constant(1.0), p)*dx
bcs = DirichletBC(V, Constant(0.0), (1,2,3,4))

solver_param = {'ksp_type': 'cg', 'pc_type': 'gamg'}
solve(a==l, q_h, bcs=bcs, solver_parameters=solver_param)

a = inner(grad(q), grad(p))*dx
solve(a==l, q_hh, bcs=bcs, solver_parameters=solver_param)

fig, ax = plt.subplots(1, 3)

tripcolor(u, axes=ax[0])
tripcolor(q_h, axes=ax[1])
tripcolor(assemble(q_hh-q_h), axes=ax[2])

plt.show()
