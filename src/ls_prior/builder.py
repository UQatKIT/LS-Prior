"""."""

from dataclasses import dataclass
from numbers import Real
from typing import Annotated

import dolfinx as dlx
import numpy as np
from beartype.vale import Is
from dolfinx.fem import petsc
from petsc4py import PETSc

from . import components, fem, prior


# ==================================================================================================
@dataclass
class BilaplacianPriorSettings:
    """_summary_."""

    mesh: dlx.mesh.Mesh
    mean_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    kappa: Annotated[Real, Is[lambda x: x > 0]]
    tau: Annotated[Real, Is[lambda x: x > 0]]
    robin_const: Real | None = None
    seed: Real = 0
    fe_data: tuple[str, Annotated[int, Is[lambda x: x > 0]]] = ("CG", 1)
    cg_relative_tolerance: Annotated[Real, Is[lambda x: x > 0]] | None = None
    cg_absolute_tolerance: Annotated[Real, Is[lambda x: x > 0]] | None = None
    cg_max_iterations: Annotated[int, Is[lambda x: x > 0]] | None = None
    amg_relative_tolerance: Annotated[Real, Is[lambda x: x > 0]] | None = None
    amg_absolute_tolerance: Annotated[Real, Is[lambda x: x > 0]] | None = None
    amg_max_iterations: Annotated[int, Is[lambda x: x > 0]] | None = None


# ==================================================================================================
class BilaplacianPriorBuilder:
    """_summary_."""

    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: BilaplacianPriorSettings) -> None:
        """_summary_.

        Args:
            settings (BilaplacianPriorSettings): _description_
        """
        self._mesh = settings.mesh
        self._mean_vector = settings.mean_vector
        self._kappa = settings.kappa
        self._tau = settings.tau
        self._robin_const = settings.robin_const
        self._seed = settings.seed
        self._fe_data = settings.fe_data

        self._cg_solver_settings = components.InverseMatrixSolverSettings(
            solver_type=PETSc.KSP.Type.CG,
            preconditioner_type=PETSc.PC.Type.JACOBI,
            relative_tolerance=settings.cg_relative_tolerance,
            absolute_tolerance=settings.cg_absolute_tolerance,
            max_num_iterations=settings.cg_max_iterations,
        )
        self._amg_solver_settings = components.InverseMatrixSolverSettings(
            solver_type=PETSc.KSP.Type.PREONLY,
            preconditioner_type=PETSc.PC.Type.GAMG,
            relative_tolerance=settings.amg_relative_tolerance,
            absolute_tolerance=settings.amg_absolute_tolerance,
            max_num_iterations=settings.amg_max_iterations,
        )

    # ----------------------------------------------------------------------------------------------
    def build(self) -> prior.Prior:
        """_summary_.

        Returns:
            prior.Prior: _description_
        """
        mass_matrix, spde_matrix, mass_matrix_factor, converter = self._build_fem_structures()
        precision_operator_interface, covariance_operator_interface, sampling_factor_interface = (
            self._build_interface(mass_matrix, spde_matrix, mass_matrix_factor, converter)
        )
        bilaplace_prior = prior.Prior(
            self._mean_vector,
            precision_operator_interface,
            covariance_operator_interface,
            sampling_factor_interface,
            seed=0,
        )
        return bilaplace_prior

    # ----------------------------------------------------------------------------------------------
    def _build_fem_structures(self) -> tuple[PETSc.Mat, PETSc.Mat, PETSc.Mat, fem.FEMConverter]:
        """_summary_.

        Returns:
            tuple[PETSc.Mat, PETSc.Mat, PETSc.Mat, fem.FEMConverter]: _description_
        """
        function_space = dlx.fem.functionspace(self._mesh, self._fe_data)
        mass_matrix_form, spde_matrix_form = fem.generate_forms(
            function_space, self._kappa, self._tau, self._robin_const
        )
        mass_matrix = petsc.assemble_matrix(dlx.fem.form(mass_matrix_form))
        spde_matrix = petsc.assemble_matrix(dlx.fem.form(spde_matrix_form))
        mass_matrix.assemble()
        spde_matrix.assemble()
        mass_matrix_factorization = fem.FEMMatrixBlockFactorization(
            self._mesh, function_space, mass_matrix_form
        )
        mass_matrix_factor = mass_matrix_factorization.assemble()
        converter = fem.FEMConverter(function_space)

        return mass_matrix, spde_matrix, mass_matrix_factor, converter

    # ----------------------------------------------------------------------------------------------
    def _build_interface(
        self,
        mass_matrix: PETSc.Mat,
        spde_matrix: PETSc.Mat,
        mass_matrix_factor: PETSc.Mat,
        converter: fem.FEMConverter,
    ) -> tuple[
        components.InterfaceComponent, components.InterfaceComponent, components.InterfaceComponent
    ]:
        """_summary_.

        Args:
            mass_matrix (PETSc.Mat): _description_
            spde_matrix (PETSc.Mat): _description_
            mass_matrix_factor (PETSc.Mat): _description_
            converter (fem.FEMConverter): _description_

        Returns:
            tuple[components.InterfaceComponent,
                  components.InterfaceComponent,
                  components.InterfaceComponent]: _description_
        """
        mass_matrix_component = components.Matrix(mass_matrix)
        spe_matrix_component = components.Matrix(spde_matrix)
        mass_matrix_factor_component = components.Matrix(mass_matrix_factor)
        mass_matrix_inverse_component = components.InverseMatrixSolver(
            self._cg_solver_settings, mass_matrix
        )
        spde_matrix_inverse_component = components.InverseMatrixSolver(
            self._amg_solver_settings, spde_matrix
        )

        precision_operator = components.BilaplacianPrecision(
            spe_matrix_component, mass_matrix_inverse_component
        )
        covariance_operator = components.BilaplacianCovariance(
            mass_matrix_component, spde_matrix_inverse_component
        )
        sampling_factor = components.BilaplacianCovarianceFactor(
            mass_matrix_factor_component, spde_matrix_inverse_component
        )

        precision_operator_interface = components.InterfaceComponent(precision_operator, converter)
        covariance_operator_interface = components.InterfaceComponent(
            covariance_operator, converter
        )
        sampling_factor_interface = components.InterfaceComponent(
            sampling_factor, converter, convert_input_from_mesh=False, convert_output_to_mesh=True
        )

        return (
            precision_operator_interface,
            covariance_operator_interface,
            sampling_factor_interface,
        )
