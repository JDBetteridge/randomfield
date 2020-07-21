from firedrake import *
from matern import matern

import matplotlib.pyplot as plt

mesh = IcosahedralSphereMesh(1, refinement_level=4, degree=2)
V = FunctionSpace(mesh, 'CG', 2)

rand = matern(V, correlation_length=0.5)

fig, ax = plt.subplots(1, 1)

tripcolor(rand, axes=ax)

plt.show()
