import numpy as np
import dolfinx as dlx
from petsc4py import PETSc
from mpi4py import MPI


# ==================================================================================================
class PETScOperator:
    def __init__(self):
        pass

    def apply(self, vector: PETSc.Vec) -> PETSc.Vec:
        pass


# ==================================================================================================
class NumpyOperator:
    def __init__(self):
        pass

    def apply(
        self, vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        pass


# ==================================================================================================
class InverseMassMatrixSolver(PETScOperator):
    # ----------------------------------------------------------------------------------------------
    def __init__(self, solver_settings, mass_matrix: PETSc.Mat, mpi_communicator: MPI.Comm):
        self._solution_vector = mass_matrix.createVecLeft()
        self._solver = PETSc.KSP().create(mpi_communicator)
        self._solver.setOperators(mass_matrix)
        self._solver.setType(solver_settings)
        self._solver.setPC(solver_settings)
        self._solver.setInitialGuessNonzero(False)
        self._solver.setTolerances(
            rtol=solver_settings.relative_tolerance,
            atol=solver_settings.absolute_tolerance,
            max_it=solver_settings.max_num_iterations,
        )

    # ----------------------------------------------------------------------------------------------
    def apply(self, vector: PETSc.Vec) -> PETSc.Vec:
        self._solver.solve(vector, self._solution_vector)
        return self._solution_vector
