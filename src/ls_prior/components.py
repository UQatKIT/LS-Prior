from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real
from typing import Annotated

import numpy as np
from beartype.vale import Is
from mpi4py import MPI
from petsc4py import PETSc


# ==================================================================================================
class PETScComponent(ABC):
    @abstractmethod
    def apply(self, input_vector: PETSc.Vec, output_vector: PETSc.Vec) -> None:
        pass

    @abstractmethod
    def create_input_vector(self) -> PETSc.Vec:
        pass

    @abstractmethod
    def create_output_vector(self) -> PETSc.Vec:
        pass


# --------------------------------------------------------------------------------------------------
class NumpyComponent(ABC):
    @abstractmethod
    def apply(
        self, vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        pass


# ==================================================================================================
@dataclass
class InverseMatrixSolverSettings:
    solver_type: PETSc.KSP.Type
    preconditioner_type: PETSc.PC.Type
    relative_tolerance: Annotated[Real, Is[lambda x: x > 0]]
    absolute_tolerance: Annotated[Real, Is[lambda x: x > 0]]
    max_num_iterations: Annotated[int, Is[lambda x: x > 0]]


class InverseMatrixSolver(PETScComponent):
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        solver_settings: InverseMatrixSolverSettings,
        matrix: PETSc.Mat,
        mpi_communicator: MPI.Comm,
    ) -> None:
        self._matrix = matrix
        self._solver = PETSc.KSP().create(mpi_communicator)
        self._solver.setOperators(matrix)
        self._solver.setType(solver_settings.solver_type)
        self._solver.setPC(solver_settings.preconditioner_type)
        self._solver.setInitialGuessNonzero(False)
        self._solver.setTolerances(
            rtol=solver_settings.relative_tolerance,
            atol=solver_settings.absolute_tolerance,
            max_it=solver_settings.max_num_iterations,
        )

    # ----------------------------------------------------------------------------------------------
    def apply(self, input_vector: PETSc.Vec, output_vector: PETSc.Vec) -> None:
        self._solver.solve(input_vector, output_vector)

    # ----------------------------------------------------------------------------------------------
    def create_input_vector(self) -> PETSc.Vec:
        return self._matrix.createVecRight()

    # ----------------------------------------------------------------------------------------------
    def create_output_vector(self) -> PETSc.Vec:
        return self._matrix.createVecLeft()


# ==================================================================================================
class BilaplacianPrecision(NumpyComponent):
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self, spde_system_matrix: PETSc.Mat, mass_matrix_inverse: InverseMatrixSolver
    ) -> None:
        self._spde_system_matrix = spde_system_matrix
        self._mass_matrix_inverse = mass_matrix_inverse
        self._input_buffer = spde_system_matrix.createVecLeft()
        self._output_buffer = spde_system_matrix.createVecRight()
        self._temp1_buffer = spde_system_matrix.createVecRight()
        self._temp2_buffer = spde_system_matrix.createVecLeft()

    # ----------------------------------------------------------------------------------------------
    def apply(
        self, input_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        if not self._input_buffer.getSize() == input_vector.size:
            raise ValueError(
                f"Input vector size {input_vector.size} does not match "
                f"expected size {self._input_buffer.getSize()}."
            )
        self._input_buffer.setArray(input_vector)
        self._spde_system_matrix.mult(self._input_buffer, self._temp1_buffer)
        self._mass_matrix_inverse.apply(self._temp1_buffer, self._temp2_buffer)
        self._spde_system_matrix.mult(self._temp2_buffer, self._output_buffer)
        output_vector = self._output_buffer.getArray()
        return output_vector


# ==================================================================================================
class BilaplacianCovariance(NumpyComponent):
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self, mass_matrix: PETSc.Mat, spde_system_matrix_inverse: InverseMatrixSolver
    ) -> None:
        self._mass_matrix = mass_matrix
        self._spde_system_matrix_inverse = spde_system_matrix_inverse
        self._input_buffer = spde_system_matrix_inverse.create_input_vector()
        self._output_buffer = spde_system_matrix_inverse.create_output_vector()
        self._temp1_buffer = mass_matrix.createVecRight()
        self._temp2_buffer = mass_matrix.createVecLeft()

    # ----------------------------------------------------------------------------------------------
    def apply(
        self, input_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        if not self._input_buffer.getSize() == input_vector.size:
            raise ValueError(
                f"Input vector size {input_vector.size} does not match "
                f"expected size {self._input_buffer.getSize()}."
            )
        self._input_buffer.setArray(input_vector)
        self._spde_system_matrix_inverse.apply(self._input_buffer, self._temp1_buffer)
        self._mass_matrix.mult(self._temp1_buffer, self._temp2_buffer)
        self._spde_system_matrix_inverse.apply(self._temp2_buffer, self._output_buffer)
        output_vector = self._output_buffer.getArray()
        return output_vector


# ==================================================================================================
class BilaplacianCovarianceFactor(NumpyComponent):
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self, mass_matrix_factor: PETSc.Mat, spde_system_matrix_inverse: InverseMatrixSolver
    ) -> None:
        self._mass_matrix_factor = mass_matrix_factor
        self._spde_system_matrix_inverse = spde_system_matrix_inverse
        self._input_buffer = mass_matrix_factor.createVecRight()
        self._output_buffer = spde_system_matrix_inverse.create_output_vector()
        self._temp1_buffer = mass_matrix_factor.createVecLeft()

    # ----------------------------------------------------------------------------------------------
    def apply(
        self, input_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        if not self._input_buffer.getSize() == input_vector.size:
            raise ValueError(
                f"Input vector size {input_vector.size} does not match "
                f"expected size {self._input_buffer.getSize()}."
            )
        self._input_buffer.setArray(input_vector)
        self._mass_matrix_factor.mult(self._input_buffer, self._temp1_buffer)
        self._spde_system_matrix_inverse.apply(self._temp1_buffer, self._output_buffer)
        output_vector = self._output_buffer.getArray()
        return output_vector

    # ----------------------------------------------------------------------------------------------
    @property
    def input_dimension(self) -> int:
        return self._mass_matrix_factor.getSize()[1]
