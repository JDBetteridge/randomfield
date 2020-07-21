def compute_stability(self, params, branchid, u, hint=None):
    V = u.function_space()
    trial = TrialFunction(V)
    test  = TestFunction(V)
    bcs = self.boundary_conditions(V, params)
    comm = V.mesh().mpi_comm()
    F = self.residual(u, map(Constant, params), test)
    J = derivative(F, u, trial)
    # Build the LHS matrix
    A = assemble(J, bcs=bcs, mat_type="aij")
    A.force_evaluation()
    A = A.M.handle
    pc = PETSc.PC().create(comm)
    pc.setOperators(A)
    pc.setType("cholesky")
    pc.setFactorSolverType("mumps")
    pc.setUp()
    F = pc.getFactorMatrix()
    (neg, zero, pos) = F.getInertia()
