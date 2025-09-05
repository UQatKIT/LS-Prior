from dataclasses import dataclass

import dolfinx as dlx
import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc


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
    return dlx.mesh.create_unit_interval(MPI.COMM_WORLD, nx=3)


def unit_square_mesh() -> dlx.mesh.Mesh:
    """Create a 2D unit square mesh."""
    return dlx.mesh.create_unit_square(
        MPI.COMM_WORLD, nx=2, ny=2, cell_type=dlx.mesh.CellType.triangle
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


@pytest.fixture(scope="session")
def fem_converter_setup_combinations():
    mesh_1d = unit_interval_mesh()
    mesh_2d = unit_square_mesh()
    fs_1d_cg2 = cg2_space(mesh_1d)
    fs_2d_cg2 = cg2_space(mesh_2d)
    return fs_1d_cg2, fs_2d_cg2


# ==================================================================================================
@pytest.fixture(scope="session")
def precomputed_assembly_matrices() -> tuple:
    """Precomputed mass and SPDE matrices for validation."""
    mass_matrix_1d_cg1 = np.zeros((4, 4))
    mass_matrix_1d_cg2 = np.zeros((7, 7))
    spde_matrix_1d_cg1 = np.zeros((4, 4))
    spde_matrix_1d_cg2 = np.zeros((7, 7))
    mass_matrix_2d_cg1 = np.zeros((9, 9))
    mass_matrix_2d_cg2 = np.zeros((25, 25))
    spde_matrix_2d_cg1 = np.zeros((9, 9))
    spde_matrix_2d_cg2 = np.zeros((25, 25))
    return [
        PrecomputedAssemblyMatrices(mass_matrix_1d_cg1, spde_matrix_1d_cg1),
        PrecomputedAssemblyMatrices(mass_matrix_1d_cg2, spde_matrix_1d_cg2),
        PrecomputedAssemblyMatrices(mass_matrix_2d_cg1, spde_matrix_2d_cg1),
        PrecomputedAssemblyMatrices(mass_matrix_2d_cg2, spde_matrix_2d_cg2),
    ]


@pytest.fixture(scope="session")
def precomputed_converter_vectors():
    input_vertex_1d = np.zeros((4,), dtype=np.float64)
    dof_values_1d = np.zeros((7,), dtype=np.float64)
    dof_vector_1d = PETSc.Vec().createWithArray(dof_values_1d, comm=MPI.COMM_WORLD)
    output_vertex_1d = np.zeros((4,), dtype=np.float64)
    input_vertex_2d = np.zeros((9,), dtype=np.float64)
    dof_values_2d = np.zeros((25,), dtype=np.float64)
    dof_vector_2d = PETSc.Vec().createWithArray(dof_values_2d, comm=MPI.COMM_WORLD)
    output_vertex_2d = np.zeros((9,), dtype=np.float64)
    return [
        PrecomputedConverterVectors(input_vertex_1d, dof_vector_1d, output_vertex_1d),
        PrecomputedConverterVectors(input_vertex_2d, dof_vector_2d, output_vertex_2d),
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
    robin_const = 1.0
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
    fem_converter_setup_combinations,
    precomputed_converter_vectors,
):
    setups = []
    for function_space, expected_results in zip(
        fem_converter_setup_combinations, precomputed_converter_vectors, strict=True
    ):
        setups.append(
            FEMConverterSetup(
                function_space=function_space,
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
NUM_FEM_SETUPS = 4
NUM_FEM_CONVERTER_SETUPS = 2


@pytest.fixture(params=list(range(NUM_FEM_SETUPS)))
def parametrized_matrix_assembly_setup(request, matrix_assembly_setup: MatrixAssemblySetup):
    return matrix_assembly_setup[request.param]


@pytest.fixture(params=list(range(NUM_FEM_CONVERTER_SETUPS)))
def parametrized_fem_converter_setup(request, fem_converter_setup: FEMConverterSetup):
    return fem_converter_setup[request.param]


@pytest.fixture(params=list(range(NUM_FEM_SETUPS)))
def parametrized_factorization_assembler_setup(
    request, factorization_assembler_setup: FactorizationAssemblerSetup
):
    return factorization_assembler_setup[request.param]
