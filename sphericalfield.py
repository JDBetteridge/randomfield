from firedrake import *
from matern import matern

import matplotlib.pyplot as plt

mesh = IcosahedralSphereMesh(1, refinement_level=5, degree=2)
V = FunctionSpace(mesh, 'CG', 2)

rand = matern(V, mean=5, variance=7, smoothness=3, correlation_length=0.5)

fig, ax = plt.subplots(1, 1)
cb = tripcolor(rand, axes=ax)
fig.colorbar(cb, ax=ax)

plt.show()
