from dataclasses import dataclass, field
from numbers import Real
from typing import Annotated

import dolfinx as dlx
import numpy as np
from beartype.vale import Is
from mpi4py import MPI
from petsc4py import PETSc

from . import components, fem, prior


# ==================================================================================================
@dataclass
class BilaplacianPriorSettings:
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
    mpi_communicator: MPI.Comm = field(default_factory=lambda: MPI.COMM_WORLD)


# ==================================================================================================
class BilaplacianPriorBuilder:
    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: BilaplacianPriorSettings):
        self._mesh = settings.mesh
        self._mean_vector = settings.mean_vector
        self._kappa = settings.kappa
        self._tau = settings.tau
        self._robin_const = settings.robin_const
        self._seed = settings.seed
        self._fe_data = settings.fe_data
        self._mpi_communicator = settings.mpi_communicator

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
        fem_handler = fem.FEMHandler(self._mesh, self._fe_data)
        mass_matrix_form, spde_matrix_form = fem_handler.generate_forms(self._kappa, self._tau)
        mass_matrix = fem_handler.assemble_matrix(mass_matrix_form)
        spde_matrix = fem_handler.assemble_matrix(spde_matrix_form)
        mass_matrix_factorization = fem.FEMMatrixBlockFactorization(
            self._mesh, fem_handler.function_space, mass_matrix_form
        )
        mass_matrix_factor = mass_matrix_factorization.assemble()
        mass_matrix_inverse = components.InverseMatrixSolver(
            self._cg_solver_settings, mass_matrix, self._mpi_communicator
        )
        spde_matrix_inverse = components.InverseMatrixSolver(
            self._amg_solver_settings, spde_matrix, self._mpi_communicator
        )
        precision_operator = components.BilaplacianPrecision(spde_matrix, mass_matrix_inverse)
        covariance_operator = components.BilaplacianCovariance(mass_matrix, spde_matrix_inverse)
        sampling_factor = components.BilaplacianCovarianceFactor(
            mass_matrix_factor, spde_matrix_inverse
        )
        num_dofs = fem_handler.function_space.dofmap.index_map.size_local
        bilaplace_prior = prior.Prior(
            self._mean_vector,
            precision_operator,
            covariance_operator,
            sampling_factor,
            dimension=num_dofs,
            seed=self._seed,
        )
        return bilaplace_prior
