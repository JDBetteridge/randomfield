from firedrake import *
from mpi4py import MPI
from randomgen import RandomGenerator, MT19937

from matern import matern
from mlmcparagen import MLMC_Solver, MLMC_Problem

rg = RandomGenerator(MT19937(12345))

class FiredrakeProblem(object):
    """

    """
    def __init__(self, V):
        q = TrialFunction(V)
        p = TestFunction(V)
        self.q_h = Function(V)
        self.q_exact = Function(V)

        # u is a placeholder function for random sample
        self.u = Function(V)
        a = inner(exp(self.u)*grad(q), grad(p))*dx
        l = inner(Constant(1.0), p)*dx
        bcs = DirichletBC(V, Constant(0.0), (1,2,3,4))

        self.problem = LinearVariationalProblem(a, l, self.q_h, bcs=bcs)
        # 'snes_view': None,
        self.solver_param = {'ksp_type': 'cg', 'pc_type': 'gamg'}
        self.solver = LinearVariationalSolver(self.problem, solver_parameters=self.solver_param)
        self.solver.solve()

    def solve(self, sample):
        self.u.assign(sample)
        self.solver.solve()
        return assemble(dot(self.q_h, self.q_h) * dx)

def sampler(V_f, V_c):
    u = matern(V_f, mean=1, variance=0.2, correlation_length=0.1, lognormal=True)
    sample_f = Function(V_f)
    sample_c = None

    if V_c is not None:
        sample_c = Function(V_c)
        inject(sample_f, sample_c)

    return sample_f, sample_c

def level_maker(finelevel, coarselevel, comm=MPI.COMM_WORLD):
    coarse_mesh = UnitCubeMesh(20, 20, 20)
    hierarchy = MeshHierarchy(coarse_mesh, finelevel, 1)
    V_f = FunctionSpace(hierarchy[finelevel], "CG", 2)
    if coarselevel < 0:
        return V_f, None
    else:
        V_c = FunctionSpace(hierarchy[coarselevel], "CG", 2)
        return V_f, V_c

# Levels and repetitions
levels = 3
repetitions = [100, 50, 10]
MLMCprob = MLMC_Problem(FiredrakeProblem, sampler, level_maker)
MLMCsolv = MLMC_Solver(MLMCprob, levels, repetitions)
estimate = MLMCsolv.solve()

print(estimate)
