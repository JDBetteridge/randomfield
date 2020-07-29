import ufl
from firedrake import *

mesh = UnitSquareMesh(1, 1)

Vhat = FunctionSpace(mesh, 'CG', 2)
broken_elements = ufl.MixedElement([ufl.BrokenElement(Vi.ufl_element()) for Vi in Vhat])
V = FunctionSpace(mesh, broken_elements)

u = TrialFunction(V)
v = TestFunction(V)

mass = inner(u,v)*dx
breakpoint()
M = assemble(mass).M
