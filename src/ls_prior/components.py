from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real
from typing import Annotated

import numpy as np
from beartype.vale import Is
from mpi4py import MPI
from petsc4py import PETSc

from . import fem


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

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        pass

    def _check_dimensions(self, input_vector: PETSc.Vec, output_vector: PETSc.Vec) -> None:
        if not input_vector.getSize() == self.shape[1]:
            raise ValueError(
                f"Input vector size {input_vector.getSize()} does not match "
                f"expected size {self.shape[1]}."
            )
        if not output_vector.getSize() == self.shape[0]:
            raise ValueError(
                f"Output vector size {output_vector.getSize()} does not match "
                f"expected size {self.shape[0]}."
            )


# ==================================================================================================
class InterfaceComponent:
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        component: PETScComponent,
        converter: fem.FEMConverter,
        convert_input_from_mesh: bool = True,
        convert_output_to_mesh: bool = True,
    ) -> None:
        self._component = component
        self._converter = converter
        self._convert_input_from_mesh = convert_input_from_mesh
        self._convert_output_to_mesh = convert_output_to_mesh
        self._input_buffer = component.create_input_vector()
        self._output_buffer = component.create_output_vector()

    # ----------------------------------------------------------------------------------------------
    def apply(
        self, input_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        input_copy = input_vector.copy()

        # Convert input from vertex-based to function space DoFs
        if self._convert_input_from_mesh:
            input_petsc = self._converter.convert_vertex_values_to_dofs(input_copy)
        else:
            if not self._input_buffer.getSize() == input_copy.shape[0]:
                raise ValueError(
                    f"Input vector size {input_copy.shape[0]} does not match "
                    f"expected size {self._input_buffer.getSize()}."
                )
            self._input_buffer.setArray(input_copy)
            self._input_buffer.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )
            input_petsc = self._input_buffer

        # Apply component transformation
        self._component.apply(input_petsc, self._output_buffer)

        # Convert result from function space DoFs to vertex-based
        if self._convert_output_to_mesh:
            output_vector = self._converter.convert_dofs_to_vertex_values(self._output_buffer)
        else:
            output_vector = self._output_buffer.getArray()

        return output_vector.copy()

    # ----------------------------------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        if self._convert_input_from_mesh:
            input_size = self._converter.vertex_space_dim
        else:
            input_size = self._component.shape[1]

        if self._convert_output_to_mesh:
            output_size = self._converter.vertex_space_dim
        else:
            output_size = self._component.shape[0]

        return output_size, input_size


# ==================================================================================================
@dataclass
class InverseMatrixSolverSettings:
    solver_type: str
    preconditioner_type: str
    relative_tolerance: Annotated[Real, Is[lambda x: x > 0]] | None
    absolute_tolerance: Annotated[Real, Is[lambda x: x > 0]] | None
    max_num_iterations: Annotated[int, Is[lambda x: x > 0]] | None


class InverseMatrixSolver(PETScComponent):
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self, solver_settings: InverseMatrixSolverSettings, petsc_matrix: PETSc.Mat
    ) -> None:
        self._petsc_matrix = petsc_matrix
        self._solver = PETSc.KSP().create(petsc_matrix.comm)
        self._solver.setOperators(petsc_matrix)
        self._solver.setType(solver_settings.solver_type)
        preconditioner = self._solver.getPC()
        preconditioner.setType(solver_settings.preconditioner_type)
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
        return self._petsc_matrix.createVecRight()

    # ----------------------------------------------------------------------------------------------
    def create_output_vector(self) -> PETSc.Vec:
        return self._petsc_matrix.createVecLeft()

    # ----------------------------------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return self._petsc_matrix.getSize()


# ==================================================================================================
class Matrix(PETScComponent):
    def __init__(self, petsc_matrix: PETSc.Mat):
        self._petsc_matrix = petsc_matrix

    # ----------------------------------------------------------------------------------------------
    def apply(self, input_vector: PETSc.Vec, output_vector: PETSc.Vec) -> None:
        self._petsc_matrix.mult(input_vector, output_vector)

    # ----------------------------------------------------------------------------------------------
    def create_input_vector(self) -> PETSc.Vec:
        return self._petsc_matrix.createVecRight()

    # ----------------------------------------------------------------------------------------------
    def create_output_vector(self) -> PETSc.Vec:
        return self._petsc_matrix.createVecLeft()

    # ----------------------------------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return self._petsc_matrix.getSize()


# ==================================================================================================
class BilaplacianPrecision(PETScComponent):
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self, spde_system_matrix: Matrix, mass_matrix_inverse: InverseMatrixSolver
    ) -> None:
        self._spde_system_matrix = spde_system_matrix
        self._mass_matrix_inverse = mass_matrix_inverse
        self._temp1_buffer = spde_system_matrix.create_output_vector()
        self._temp2_buffer = spde_system_matrix.create_input_vector()

    # ----------------------------------------------------------------------------------------------
    def apply(self, input_vector: PETSc.Vec, output_vector: PETSc.Vec) -> None:
        self._check_dimensions(input_vector, output_vector)
        self._spde_system_matrix.apply(input_vector, self._temp1_buffer)
        self._mass_matrix_inverse.apply(self._temp1_buffer, self._temp2_buffer)
        self._spde_system_matrix.apply(self._temp2_buffer, output_vector)

    # ----------------------------------------------------------------------------------------------
    def create_input_vector(self) -> PETSc.Vec:
        return self._spde_system_matrix.create_input_vector()

    # ----------------------------------------------------------------------------------------------
    def create_output_vector(self) -> PETSc.Vec:
        return self._spde_system_matrix.create_output_vector()

    # ----------------------------------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return self._spde_system_matrix.shape


# ==================================================================================================
class BilaplacianCovariance(PETScComponent):
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self, mass_matrix: Matrix, spde_system_matrix_inverse: InverseMatrixSolver
    ) -> None:
        self._mass_matrix = mass_matrix
        self._spde_system_matrix_inverse = spde_system_matrix_inverse
        self._temp1_buffer = mass_matrix.create_input_vector()
        self._temp2_buffer = mass_matrix.create_output_vector()

    # ----------------------------------------------------------------------------------------------
    def apply(self, input_vector: PETSc.Vec, output_vector: PETSc.Vec) -> None:
        self._check_dimensions(input_vector, output_vector)
        self._spde_system_matrix_inverse.apply(input_vector, self._temp1_buffer)
        self._mass_matrix.apply(self._temp1_buffer, self._temp2_buffer)
        self._spde_system_matrix_inverse.apply(self._temp2_buffer, output_vector)

    # ----------------------------------------------------------------------------------------------
    def create_input_vector(self) -> PETSc.Vec:
        return self._spde_system_matrix_inverse.create_input_vector()

    # ----------------------------------------------------------------------------------------------
    def create_output_vector(self) -> PETSc.Vec:
        return self._spde_system_matrix_inverse.create_output_vector()

    # ----------------------------------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return self._spde_system_matrix_inverse.shape


# ==================================================================================================
class BilaplacianCovarianceFactor(PETScComponent):
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self, mass_matrix_factor: Matrix, spde_system_matrix_inverse: InverseMatrixSolver
    ) -> None:
        self._mass_matrix_factor = mass_matrix_factor
        self._spde_system_matrix_inverse = spde_system_matrix_inverse
        self._temp_buffer = mass_matrix_factor.create_output_vector()

    # ----------------------------------------------------------------------------------------------
    def apply(self, input_vector: PETSc.Vec, output_vector: PETSc.Vec) -> None:
        self._check_dimensions(input_vector, output_vector)
        self._mass_matrix_factor.apply(input_vector, self._temp_buffer)
        self._spde_system_matrix_inverse.apply(self._temp_buffer, output_vector)

    # ----------------------------------------------------------------------------------------------
    def create_input_vector(self) -> PETSc.Vec:
        return self._mass_matrix_factor.create_input_vector()

    # ----------------------------------------------------------------------------------------------
    def create_output_vector(self) -> PETSc.Vec:
        return self._spde_system_matrix_inverse.create_output_vector()

    # ----------------------------------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return self._spde_system_matrix_inverse.shape[0], self._mass_matrix_factor.shape[1]
