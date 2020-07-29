from firedrake import *
from matern import matern

import matplotlib.pyplot as plt

mesh = UnitCubeMesh(20, 20, 20)
V = FunctionSpace(mesh, 'CG', 2)

rand = matern(V, mean=5, variance=7, smoothness=2, correlation_length=0.5)

outfile = File('randomcube.pvd')
outfile.write(rand)
