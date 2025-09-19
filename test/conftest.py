from dataclasses import dataclass

import dolfinx as dlx
import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

# ==================================================================================================
MPI_COMMUNICATOR = MPI.COMM_WORLD
NUM_FEM_SETUPS = 4
FEM_SETUP_IDS = ["1d_cg1", "1d_cg2", "2d_cg1", "2d_cg2"]
FEM_MATRIX_DATA = "test/data/fem_matrices.npz"
FEM_CONVERTER_DATA = "test/data/fem_converter_vectors.npz"


# ==================================================================================================
@dataclass
class PrecomputedAssemblyMatrices:
    mass_matrix: np.ndarray
    spde_matrix: np.ndarray


@dataclass
class PrecomputedConverterVectors:
    input_vertex_values: np.ndarray
    dof_values: PETSc.Vec
    output_vertex_values: np.ndarray


# ==================================================================================================
@dataclass
class FEMSpaceSetup:
    mesh: dlx.mesh.Mesh
    function_space: dlx.fem.FunctionSpace


@dataclass
class MatrixAssemblySetup:
    function_space: dlx.fem.FunctionSpace
    kappa: float
    tau: float
    robin_const: float
    expected_mass_matrix: np.ndarray
    expected_spde_matrix: np.ndarray


@dataclass
class FEMConverterSetup:
    function_space: dlx.fem.FunctionSpace
    input_vertex_values: np.ndarray
    expected_dof_values: PETSc.Vec
    expected_output_vertex_values: np.ndarray


@dataclass
class FactorizationAssemblerSetup:
    fem_setup: FEMSpaceSetup
    expected_mass_matrix: np.ndarray


@dataclass
class MatrixComponentSetup:
    mass_matrix_array: np.ndarray
    spde_matrix_array: np.ndarray
    mass_matrix_petsc: PETSc.Mat
    spde_matrix_petsc: PETSc.Mat
    input_array: np.ndarray
    input_vector: PETSc.Vec


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


# ==================================================================================================
@pytest.fixture(scope="session")
def precomputed_assembly_matrices() -> list[PrecomputedAssemblyMatrices]:
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


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def precomputed_converter_vectors() -> list[PrecomputedConverterVectors]:
    femconverter_vectors = np.load(FEM_CONVERTER_DATA)

    input_vertex_1d = femconverter_vectors["input_vertex_1d"]
    input_vertex_2d = femconverter_vectors["input_vertex_2d"]
    dof_vector_1d_cg1 = femconverter_vectors["dof_values_1d_cg1"]
    dof_vector_1d_cg2 = femconverter_vectors["dof_values_1d_cg2"]
    dof_vector_2d_cg1 = femconverter_vectors["dof_values_2d_cg1"]
    dof_vector_2d_cg2 = femconverter_vectors["dof_values_2d_cg2"]
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
def fem_setup_combinations() -> list[FEMSpaceSetup]:
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


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def matrix_assembly_setups(
    fem_setup_combinations: list[FEMSpaceSetup],
    precomputed_assembly_matrices: list[PrecomputedAssemblyMatrices],
) -> list[MatrixAssemblySetup]:
    kappa = 1.0
    tau = 1.0
    robin_const = None
    setups = []
    for fem_setup, precomputed_results in zip(
        fem_setup_combinations, precomputed_assembly_matrices, strict=True
    ):
        setups.append(
            MatrixAssemblySetup(
                function_space=fem_setup.function_space,
                kappa=kappa,
                tau=tau,
                robin_const=robin_const,
                expected_mass_matrix=precomputed_results.mass_matrix,
                expected_spde_matrix=precomputed_results.spde_matrix,
            )
        )
    return setups


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def fem_converter_setups(
    fem_setup_combinations: list[FEMSpaceSetup],
    precomputed_converter_vectors: list[PrecomputedConverterVectors],
) -> list[FEMConverterSetup]:
    setups = []
    for fem_setup, precomputed_results in zip(
        fem_setup_combinations, precomputed_converter_vectors, strict=True
    ):
        setups.append(
            FEMConverterSetup(
                function_space=fem_setup.function_space,
                input_vertex_values=precomputed_results.input_vertex_values,
                expected_dof_values=precomputed_results.dof_values,
                expected_output_vertex_values=precomputed_results.output_vertex_values,
            )
        )
    return setups


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def factorization_assembler_setups(
    fem_setup_combinations: list[FEMSpaceSetup],
    precomputed_assembly_matrices: list[PrecomputedAssemblyMatrices],
) -> list[FactorizationAssemblerSetup]:
    setups = []
    for fem_setup, precomputed_results in zip(
        fem_setup_combinations, precomputed_assembly_matrices, strict=True
    ):
        setups.append(
            FactorizationAssemblerSetup(
                fem_setup=fem_setup,
                expected_mass_matrix=precomputed_results.expected_mass_matrix,
            )
        )
    return setups


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def matrix_component_setup(precomputed_assembly_matrices: list[PrecomputedAssemblyMatrices]):
    matrix_representations = []
    rng = np.random.default_rng(0)

    for assembly_matrices in precomputed_assembly_matrices:
        mass_matrix_array = assembly_matrices.mass_matrix
        spde_matrix_array = assembly_matrices.spde_matrix
        mass_matrix_petsc = PETSc.Mat().createAIJ(
            size=mass_matrix_array.shape, comm=MPI_COMMUNICATOR
        )
        spde_matrix_petsc = PETSc.Mat().createAIJ(
            size=spde_matrix_array.shape, comm=MPI_COMMUNICATOR
        )
        for petsc_matrix, array in (
            (mass_matrix_petsc, mass_matrix_array),
            (spde_matrix_petsc, spde_matrix_array),
        ):
            petsc_matrix.setUp()
            row_inds = np.arange(array.shape[0], dtype=np.int32)
            col_inds = np.arange(array.shape[1], dtype=np.int32)
            petsc_matrix.setValues(row_inds, col_inds, array)
            petsc_matrix.assemble()

        input_array = rng.random(array.shape[1])
        input_vector = PETSc.Vec().createWithArray(input_array, comm=MPI_COMMUNICATOR)
        matrix_representation = MatrixComponentSetup(
            mass_matrix_array,
            spde_matrix_array,
            mass_matrix_petsc,
            spde_matrix_petsc,
            input_array,
            input_vector,
        )
        matrix_representations.append(matrix_representation)

    return matrix_representations


# ==================================================================================================
@pytest.fixture(params=list(range(NUM_FEM_SETUPS)), ids=FEM_SETUP_IDS)
def parametrized_matrix_assembly_setup(
    request: pytest.FixtureRequest, matrix_assembly_setups: list[MatrixAssemblySetup]
) -> MatrixAssemblySetup:
    return matrix_assembly_setups[request.param]


@pytest.fixture(params=list(range(NUM_FEM_SETUPS)), ids=FEM_SETUP_IDS)
def parametrized_fem_converter_setup(
    request: pytest.FixtureRequest, fem_converter_setups: list[FEMConverterSetup]
) -> FEMConverterSetup:
    return fem_converter_setups[request.param]


@pytest.fixture(params=list(range(NUM_FEM_SETUPS)), ids=FEM_SETUP_IDS)
def parametrized_factorization_assembler_setup(
    request: pytest.FixtureRequest,
    factorization_assembler_setups: list[FactorizationAssemblerSetup],
) -> FactorizationAssemblerSetup:
    return factorization_assembler_setups[request.param]


@pytest.fixture(params=list(range(NUM_FEM_SETUPS)), ids=FEM_SETUP_IDS)
def parametrized_matrix_representation(
    request: pytest.FixtureRequest,
    matrix_component_setup: list[MatrixComponentSetup],
) -> MatrixComponentSetup:
    return matrix_component_setup[request.param]
