from firedrake import *
from matern import matern

import matplotlib.pyplot as plt

mesh = UnitSquareMesh(50, 50)
V = FunctionSpace(mesh, 'CG', 2)

randomlist = []
for l in [1, 0.5, 0.25, 0.125, 0.06125, 0.0000001]:
    randomlist.append(matern(V, correlation_length=l))

fig, ax = plt.subplots(2, 3)

for ii, axes in enumerate(ax.flatten()):
    tripcolor(randomlist[ii], axes=axes)

plt.show()
