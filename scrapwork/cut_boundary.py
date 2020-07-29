from firedrake import *

import matplotlib.pyplot as plt

mesh = RectangleMesh(30, 30, 3, 3)
mesh.coordinates.dat.data[:, :] = mesh.coordinates.dat.data - 1

submesh = RectangleMesh(10, 10, 1, 1)

V = FunctionSpace(mesh, 'CG', 1)
f = Function(V)
x, y = SpatialCoordinate(mesh)
f.interpolate(exp(- (x - 1)**2 - (y - 1)**2))

W = FunctionSpace(submesh, V.ufl_element())
g = Function(W)
g.dat.data[:] = f.at(submesh.coordinates.dat.data)

fig, ax = plt.subplots(1, 2)

# ~ triplot(mesh, axes=ax)
# ~ triplot(submesh, axes=ax, interior_kw={'edgecolor' : 'r'})

tripcolor(f, axes=ax[0])
tripcolor(g, axes=ax[1])

plt.show()
