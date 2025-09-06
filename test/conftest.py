from dataclasses import dataclass

import dolfinx as dlx
import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

# ==================================================================================================
MPI_COMMUNICATOR = MPI.COMM_WORLD
NUM_FEM_SETUPS = 4
FEM_MATRIX_DATA = "test/data/fem_matrices.npz"
FEM_CONVERTER_DATA = "test/data/fem_converter_vectors.npz"


# ==================================================================================================
@dataclass
class FEMSpaceSetup:
    mesh: dlx.mesh.Mesh
    function_space: dlx.fem.FunctionSpace


@dataclass
class PrecomputedAssemblyMatrices:
    mass_matrix: np.ndarray
    spde_matrix: np.ndarray


@dataclass
class PrecomputedConverterVectors:
    input_vertex_values: np.ndarray
    dof_values: PETSc.Vec
    output_vertex_values: np.ndarray


@dataclass
class MatrixAssemblySetup:
    function_space: dlx.fem.FunctionSpace
    kappa: float
    tau: float
    robin_const: float
    mass_matrix: np.ndarray
    spde_matrix: np.ndarray


@dataclass
class FEMConverterSetup:
    function_space: dlx.fem.FunctionSpace
    input_vertex_values: np.ndarray
    dof_values: PETSc.Vec
    output_vertex_values: np.ndarray


@dataclass
class FactorizationAssemblerSetup:
    mesh: dlx.mesh.Mesh
    function_space: dlx.fem.FunctionSpace
    mass_matrix: np.ndarray


# ==================================================================================================
def unit_interval_mesh() -> dlx.mesh.Mesh:
    """Create a 1D unit interval mesh."""
    return dlx.mesh.create_unit_interval(MPI_COMMUNICATOR, nx=3)


def unit_square_mesh() -> dlx.mesh.Mesh:
    """Create a 2D unit square mesh."""
    return dlx.mesh.create_unit_square(
        MPI_COMMUNICATOR, nx=2, ny=2, cell_type=dlx.mesh.CellType.triangle
    )


def cg1_space(mesh: dlx.mesh.Mesh) -> dlx.fem.FunctionSpace:
    """P1 continuous Galerkin function space."""
    return dlx.fem.functionspace(mesh, ("Lagrange", 1))


def cg2_space(mesh: dlx.mesh.Mesh) -> dlx.fem.FunctionSpace:
    """P2 continuous Galerkin function space."""
    return dlx.fem.functionspace(mesh, ("Lagrange", 2))


@pytest.fixture(scope="session")
def fem_setup_combinations():
    mesh_1d = unit_interval_mesh()
    mesh_2d = unit_square_mesh()
    fs_1d_cg1 = cg1_space(mesh_1d)
    fs_1d_cg2 = cg2_space(mesh_1d)
    fs_2d_cg1 = cg1_space(mesh_2d)
    fs_2d_cg2 = cg2_space(mesh_2d)
    return [
        FEMSpaceSetup(mesh_1d, fs_1d_cg1),
        FEMSpaceSetup(mesh_1d, fs_1d_cg2),
        FEMSpaceSetup(mesh_2d, fs_2d_cg1),
        FEMSpaceSetup(mesh_2d, fs_2d_cg2),
    ]


# ==================================================================================================
@pytest.fixture(scope="session")
def precomputed_assembly_matrices() -> tuple:
    """Precomputed mass and SPDE matrices for validation."""
    fem_matrices = np.load(FEM_MATRIX_DATA)

    mass_matrix_1d_cg1 = fem_matrices["mass_matrix_1d_cg1"]
    mass_matrix_1d_cg2 = fem_matrices["mass_matrix_1d_cg2"]
    spde_matrix_1d_cg1 = fem_matrices["spde_matrix_1d_cg1"]
    spde_matrix_1d_cg2 = fem_matrices["spde_matrix_1d_cg2"]
    mass_matrix_2d_cg1 = fem_matrices["mass_matrix_2d_cg1"]
    mass_matrix_2d_cg2 = fem_matrices["mass_matrix_2d_cg2"]
    spde_matrix_2d_cg1 = fem_matrices["spde_matrix_2d_cg1"]
    spde_matrix_2d_cg2 = fem_matrices["spde_matrix_2d_cg2"]

    return [
        PrecomputedAssemblyMatrices(mass_matrix_1d_cg1, spde_matrix_1d_cg1),
        PrecomputedAssemblyMatrices(mass_matrix_1d_cg2, spde_matrix_1d_cg2),
        PrecomputedAssemblyMatrices(mass_matrix_2d_cg1, spde_matrix_2d_cg1),
        PrecomputedAssemblyMatrices(mass_matrix_2d_cg2, spde_matrix_2d_cg2),
    ]


@pytest.fixture(scope="session")
def precomputed_converter_vectors():
    femconverter_vectors = np.load(FEM_CONVERTER_DATA)

    input_vertex_1d = femconverter_vectors["input_vertex_1d"]
    input_vertex_2d = femconverter_vectors["input_vertex_2d"]

    dof_vector_1d_cg1 = PETSc.Vec().createWithArray(
        femconverter_vectors["dof_values_1d_cg1"], comm=MPI_COMMUNICATOR
    )
    dof_vector_1d_cg2 = PETSc.Vec().createWithArray(
        femconverter_vectors["dof_values_1d_cg2"], comm=MPI_COMMUNICATOR
    )
    dof_vector_2d_cg1 = PETSc.Vec().createWithArray(
        femconverter_vectors["dof_values_2d_cg1"], comm=MPI_COMMUNICATOR
    )
    dof_vector_2d_cg2 = PETSc.Vec().createWithArray(
        femconverter_vectors["dof_values_2d_cg2"], comm=MPI_COMMUNICATOR
    )

    output_vertex_1d_cg1 = femconverter_vectors["output_vertex_1d_cg1"]
    output_vertex_1d_cg2 = femconverter_vectors["output_vertex_1d_cg2"]
    output_vertex_2d_cg1 = femconverter_vectors["output_vertex_2d_cg1"]
    output_vertex_2d_cg2 = femconverter_vectors["output_vertex_2d_cg2"]

    return [
        PrecomputedConverterVectors(input_vertex_1d, dof_vector_1d_cg1, output_vertex_1d_cg1),
        PrecomputedConverterVectors(input_vertex_1d, dof_vector_1d_cg2, output_vertex_1d_cg2),
        PrecomputedConverterVectors(input_vertex_2d, dof_vector_2d_cg1, output_vertex_2d_cg1),
        PrecomputedConverterVectors(input_vertex_2d, dof_vector_2d_cg2, output_vertex_2d_cg2),
    ]


# ==================================================================================================
@pytest.fixture(scope="session")
def matrix_assembly_setup(
    fem_setup_combinations,
    precomputed_assembly_matrices,
) -> list[MatrixAssemblySetup]:
    """Combine FEM setups with precomputed matrices for testing."""
    kappa = 1.0
    tau = 1.0
    robin_const = None
    setups = []
    for fem_setup, expected_results in zip(
        fem_setup_combinations, precomputed_assembly_matrices, strict=True
    ):
        setups.append(
            MatrixAssemblySetup(
                function_space=fem_setup.function_space,
                kappa=kappa,
                tau=tau,
                robin_const=robin_const,
                mass_matrix=expected_results.mass_matrix,
                spde_matrix=expected_results.spde_matrix,
            )
        )
    return setups


@pytest.fixture(scope="session")
def fem_converter_setup(
    fem_setup_combinations,
    precomputed_converter_vectors,
):
    setups = []
    for fem_setup, expected_results in zip(
        fem_setup_combinations, precomputed_converter_vectors, strict=True
    ):
        setups.append(
            FEMConverterSetup(
                function_space=fem_setup.function_space,
                input_vertex_values=expected_results.input_vertex_values,
                dof_values=expected_results.dof_values,
                output_vertex_values=expected_results.output_vertex_values,
            )
        )
    return setups


@pytest.fixture(scope="session")
def factorization_assembler_setup(fem_setup_combinations, precomputed_assembly_matrices) -> list:
    setups = []
    for fem_setup, expected_results in zip(
        fem_setup_combinations, precomputed_assembly_matrices, strict=True
    ):
        setups.append(
            FactorizationAssemblerSetup(
                mesh=fem_setup.mesh,
                function_space=fem_setup.function_space,
                mass_matrix=expected_results.mass_matrix,
            )
        )
    return setups


# ==================================================================================================
@pytest.fixture(params=list(range(NUM_FEM_SETUPS)))
def parametrized_matrix_assembly_setup(request, matrix_assembly_setup: MatrixAssemblySetup):
    return matrix_assembly_setup[request.param]


@pytest.fixture(params=list(range(NUM_FEM_SETUPS)))
def parametrized_fem_converter_setup(request, fem_converter_setup: FEMConverterSetup):
    return fem_converter_setup[request.param]


@pytest.fixture(params=list(range(NUM_FEM_SETUPS)))
def parametrized_factorization_assembler_setup(
    request, factorization_assembler_setup: FactorizationAssemblerSetup
):
    return factorization_assembler_setup[request.param]
