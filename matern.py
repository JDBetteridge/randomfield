from firedrake import *

from numpy.linalg import cholesky
from scipy.special import gamma
from firedrake.petsc import PETSc

import ufl
import numpy as np

## Extrct the PETSc matrix
def get_PETSC_matrix(form, **kwargs):
    operator = assemble(form, **kwargs).M.handle
    return operator

## Convert to numpy matrix
def get_np_matrix(form, **kwargs):
    L = get_PETSC_matrix(form, **kwargs)

    size = L.getSize()
    Lij = PETSc.Mat()
    L.convert('aij', Lij)

    Lnp = np.array(Lij.getValues(range(size[0]), range(size[1])))
    return Lnp

def matern(V, variance=1, smoothness=1, correlation_length=1, rg=None):
    d = V.mesh().topological_dimension()
    nu = smoothness
    sigma = np.sqrt(variance)
    lambd = correlation_length

    k = (nu + d/2)/2
    assert k==1
    kappa = np.sqrt(8)/lambd
    sigma_hat = gamma(nu)*nu**(d/2)
    sigma_hat /= gamma(nu + d/2)
    sigma_hat *= (2/np.pi)**(d/2)
    sigma_hat *= lambd**(-d)
    eta = sigma/sigma_hat

    if rg is None:
        pcg = PCG64(seed=100)
        rg = RandomGenerator(pcg)

    Wnoise = rg.normal(V, 0.0, 1.0)
    HWnoise = Function(V)

    u = TrialFunction(V)
    v = TestFunction(V)

    bcs = DirichletBC(V, 0, (1,2,3,4))

    mass = inner(u, v)*dx

    M = assemble(mass).M

    M = get_np_matrix(mass)
    H = cholesky(M)
    HWnoise.dat.data[:] = H@Wnoise.dat.data

    a = (inner(u, v) + Constant(1/(kappa**2))*inner(grad(u), grad(v)))*dx
    l = Constant(eta)*inner(HWnoise, v)*dx

    u_h = Function(V)
    solve(a == l, u_h, bcs=bcs, solver_parameters={'ksp_type': 'cg', 'pc_type': 'gamg'})

    return u_h
