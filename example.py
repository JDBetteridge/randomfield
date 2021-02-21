from firedrake import *
from matern import matern

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ~ mesh = UnitSquareMesh(50, 50)
mesh = SquareMesh(32, 32, 2)
V = FunctionSpace(mesh, 'CG', 1)
mean = 2000
var = 10000
lognormal = False

randomlist = []
for l in [1, 0.5, 0.25, 0.125, 0.06125, 0.0000001]:
    randomlist.append(matern(V,
                             mean=mean,
                             variance=var,
                             correlation_length=l,
                             lognormal=lognormal))

fig, ax = plt.subplots(2, 3)

for ii, axes in enumerate(ax.flatten()):
    if lognormal:
        vmax = np.exp(mean + 2*np.sqrt(var))
        vmin = np.exp(mean - 2*np.sqrt(var))
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmax = mean + 2*np.sqrt(var)
        vmin = mean - 2*np.sqrt(var)
        sd = np.sqrt(var)
        b = [mean + n*sd for n in range(-4, 5)]
        norm = mpl.colors.BoundaryNorm(boundaries=b, ncolors=256)
        # ~ norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
    cb = tripcolor(randomlist[ii], axes=axes, norm=norm)
    fig.colorbar(cb, ax=axes, extend='both')

plt.show()
