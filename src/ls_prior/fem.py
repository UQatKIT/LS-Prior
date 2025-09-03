"""."""

from numbers import Real
from typing import Annotated

import cffi
import dolfinx as dlx
import numpy as np
import scifem as dlx_helper
import ufl
from beartype.vale import Is
from mpi4py import MPI
from petsc4py import PETSc

ffi = cffi.FFI()


# ==================================================================================================
def generate_forms(
    function_space: dlx.fem.FunctionSpace,
    kappa: Annotated[Real, Is[lambda x: x > 0]],
    tau: Annotated[Real, Is[lambda x: x > 0]],
    robin_const: Real | None = None,
) -> tuple[ufl.Form, ufl.Form]:
    """.

    Args:
        function_space (dlx.fem.FunctionSpace): _description_
        kappa (Real): _description_
        tau (Real): _description_
        robin_const (Real | None, optional): _description_. Defaults to None.

    Returns:
        tuple[ufl.Form, ufl.Form]: _description_
    """
    trial_function = ufl.TrialFunction(function_space)
    test_function = ufl.TestFunction(function_space)
    mass_matrix_form = ufl.inner(trial_function, test_function) * ufl.dx
    stiffness_matrix_form = ufl.inner(ufl.grad(trial_function), ufl.grad(test_function)) * ufl.dx
    spde_matrix_form = kappa**2 * tau * mass_matrix_form + tau * stiffness_matrix_form
    if robin_const is not None:
        robin_boundary_form = robin_const * ufl.inner(trial_function, test_function) * ufl.ds
        spde_matrix_form += robin_boundary_form
    return mass_matrix_form, spde_matrix_form


# ==================================================================================================
class FEMConverter:
    """."""

    # ----------------------------------------------------------------------------------------------
    def __init__(self, function_space: dlx.fem.FunctionSpace) -> None:
        """_summary_.

        Args:
            function_space (dlx.fem.FunctionSpace): _description_
        """
        vertex_space = dlx.fem.functionspace(function_space.mesh, ("Lagrange", 1))
        self._dof_function = dlx.fem.Function(function_space)
        self._vertex_function = dlx.fem.Function(vertex_space)
        self.dof_space_dim = function_space.dofmap.index_map.size_local
        self.vertex_space_dim = vertex_space.dofmap.index_map.size_local
        self._vertex_to_dof_map = dlx_helper.vertex_to_dofmap(vertex_space)
        self._dof_to_vertex_map = dlx_helper.dof_to_vertexmap(vertex_space)

    # ----------------------------------------------------------------------------------------------
    def convert_vertex_values_to_dofs(
        self, vertex_values: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> PETSc.Vec:
        """_summary_.

        Args:
            vertex_values (np.ndarray[tuple[int], np.dtype[np.float64]]): _description_

        Raises:
            ValueError: _description_

        Returns:
            PETSc.Vec: _description_
        """
        if not vertex_values.shape == (self.vertex_space_dim,):
            raise ValueError(
                f"Expected vertex_values to have shape {(self.vertex_space_dim,)}, "
                f"but got {vertex_values.shape}"
            )
        self._vertex_function.x.array[:] = vertex_values[self._vertex_to_dof_map]
        self._vertex_function.x.scatter_forward()
        self._dof_function.interpolate(self._vertex_function)
        assert self._dof_function.x.array.shape == (self.dof_space_dim,), (
            f"Created PETSc vector has size {self._dof_function.x.array.shape}, "
            f"but expected {(self.dof_space_dim,)}"
        )
        return self._dof_function.x.petsc_vec

    # ----------------------------------------------------------------------------------------------
    def convert_dofs_to_vertex_values(
        self, dof_values: PETSc.Vec
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """_summary_.

        Args:
            dof_values (PETSc.Vec): _description_

        Raises:
            ValueError: _description_

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: _description_
        """
        if not dof_values.getSize() == self.dof_space_dim:
            raise ValueError(
                f"Expected dof_values to have size {self.dof_space_dim}, "
                f"but got {dof_values.getSize()}"
            )
        self._dof_function.x.array[:] = dof_values
        self._dof_function.x.scatter_forward()
        self._vertex_function.interpolate(self._dof_function)
        assert self._vertex_function.x.array.shape == (self.vertex_space_dim,), (
            f"Created PETSc vector has size {self._vertex_function.x.array.shape}, "
            f"but expected {(self.vertex_space_dim,)}"
        )
        vertex_values = self._vertex_function.x.array[self._dof_to_vertex_map]
        return vertex_values


# ==================================================================================================
class FEMMatrixBlockFactorization:
    """."""

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self, mesh: dlx.mesh.Mesh, function_space: dlx.fem.FunctionSpace, form: ufl.Form
    ) -> None:
        """_summary_.

        Args:
            mesh (dlx.mesh.Mesh): _description_
            function_space (dlx.fem.FunctionSpace): _description_
            form (ufl.Form): _description_
        """
        self._mpi_communicator = mesh.comm
        self._local_vertex_coordinates = mesh.geometry.x
        self._local_cell_vertex_indices = mesh.geometry.dofmap
        self._pproc_cell_distribution_map = mesh.topology.index_map(mesh.topology.dim)
        self._dofmap = function_space.dofmap
        self._pproc_dof_distribution_map = function_space.dofmap.index_map

        self._num_local_cells = self._pproc_cell_distribution_map.size_local
        self._num_global_cells = self._pproc_cell_distribution_map.size_global
        self._num_local_dofs = self._pproc_dof_distribution_map.size_local
        self._num_global_dofs = self._pproc_dof_distribution_map.size_global
        self._num_cell_dofs = function_space.dofmap.dof_layout.num_dofs

        self._assembly_kernel = self._init_assembly_kernel(mesh.comm, form)

    # ----------------------------------------------------------------------------------------------
    def assemble(self) -> PETSc.Mat:
        """_summary_.

        Returns:
            PETSc.Mat: _description_
        """
        block_diagonal_matrix, local_global_dof_matrix = self._set_up_petsc_mats()
        self._assemble_matrices_over_cells(block_diagonal_matrix, local_global_dof_matrix)
        block_diagonal_matrix.assemble()
        local_global_dof_matrix.assemble()
        local_global_dof_matrix.transpose()
        matrix_factorization = local_global_dof_matrix.matMult(block_diagonal_matrix)
        return matrix_factorization

    # ----------------------------------------------------------------------------------------------
    def _init_assembly_kernel(self, mpi_communicator: MPI.Comm, form: ufl.Form) -> cffi.FFI.CData:
        """_summary_.

        Args:
            mpi_communicator (MPI.Comm): _description_
            form (ufl.Form): _description_

        Returns:
            cffi.FFI.CData: _description_
        """
        form_compiled, *_ = dlx.jit.ffcx_jit(
            mpi_communicator,
            form,
            form_compiler_options={"scalar_type": PETSc.ScalarType},
        )
        compiled_kernel = getattr(
            form_compiled.form_integrals[0],
            f"tabulate_tensor_{np.dtype(PETSc.ScalarType).name}",
        )
        return compiled_kernel

    # ----------------------------------------------------------------------------------------------
    def _set_up_petsc_mats(self) -> tuple[PETSc.Mat, PETSc.Mat]:
        """_summary_.

        Returns:
            tuple[PETSc.Mat, PETSc.Mat]: _description_
        """
        block_diagonal_matrix = PETSc.Mat().createAIJ(
            [
                self._num_global_cells * self._num_cell_dofs,
                self._num_global_cells * self._num_cell_dofs,
            ],
            comm=self._mpi_communicator,
        )
        block_diagonal_matrix.setPreallocationNNZ(self._num_cell_dofs)
        block_diagonal_matrix.setUp()

        local_global_dof_matrix = PETSc.Mat().createAIJ(
            [self._num_global_cells * self._num_cell_dofs, self._num_global_dofs],
            comm=self._mpi_communicator,
        )
        local_global_dof_matrix.setPreallocationNNZ(1)
        local_global_dof_matrix.setUp()

        return block_diagonal_matrix, local_global_dof_matrix

    # ----------------------------------------------------------------------------------------------
    def _insert_in_block_diagonal_matrix(
        self,
        global_ind: np.integer,
        cell_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        block_diagonal_matrix: PETSc.Mat,
    ) -> None:
        """_summary_.

        Args:
            global_ind (np.integer): _description_
            cell_matrix (np.ndarray[tuple[int, int], np.dtype[np.float64]]): _description_
            block_diagonal_matrix (PETSc.Mat): _description_
        """
        row_col_inds = np.arange(
            global_ind * self._num_cell_dofs,
            (global_ind + 1) * self._num_cell_dofs,
            dtype=PETSc.IntType,
        )
        block_diagonal_matrix.setValues(
            row_col_inds,
            row_col_inds,
            cell_matrix,
            addv=PETSc.InsertMode.INSERT_VALUES,
        )

    # ----------------------------------------------------------------------------------------------
    def _insert_in_local_global_dof_matrix(
        self,
        global_ind: np.integer,
        global_cell_dofs: np.ndarray[tuple[int], np.dtype[np.integer]],
        local_global_dof_matrix: PETSc.Mat,
    ) -> None:
        """_summary_.

        Args:
            global_ind (np.integer): _description_
            global_cell_dofs (np.ndarray[tuple[int], np.dtype[np.integer]]): _description_
            local_global_dof_matrix (PETSc.Mat): _description_
        """
        row_inds = np.arange(
            global_ind * self._num_cell_dofs,
            (global_ind + 1) * self._num_cell_dofs,
            dtype=PETSc.IntType,
        )
        for row_ind, col_ind in zip(row_inds, global_cell_dofs, strict=True):
            local_global_dof_matrix.setValues(
                row_ind,
                col_ind,
                1.0,
                addv=PETSc.InsertMode.INSERT_VALUES,
            )

    # ----------------------------------------------------------------------------------------------
    def _assemble_matrices_over_cells(
        self, block_diagonal_matrix: PETSc.Mat, local_global_dof_matrix: PETSc.Mat
    ) -> None:
        """_summary_.

        Args:
            block_diagonal_matrix (PETSc.Mat): _description_
            local_global_dof_matrix (PETSc.Mat): _description_
        """
        local_cell_inds = np.arange(self._num_local_cells, dtype=np.int64)
        global_cell_inds = self._pproc_cell_distribution_map.local_to_global(local_cell_inds)

        for local_ind, global_ind in zip(local_cell_inds, global_cell_inds, strict=True):
            local_cell_dofs = self._dofmap.cell_dofs(local_ind)
            global_cell_dofs = self._pproc_dof_distribution_map.local_to_global(
                local_cell_dofs
            ).astype(PETSc.IntType)
            cell_vertex_inds = self._local_cell_vertex_indices[local_ind]
            cell_vertex_coordinates = self._local_vertex_coordinates[cell_vertex_inds]
            cell_matrix = np.zeros(
                (self._num_cell_dofs, self._num_cell_dofs), dtype=PETSc.ScalarType
            )
            self._assembly_kernel(
                ffi.from_buffer(cell_matrix),  # A - output matrix
                ffi.NULL,  # w - coefficient values (none for this form)
                ffi.NULL,  # c - constants (none for this form)
                ffi.from_buffer(cell_vertex_coordinates),  # coordinate_dofs
                ffi.NULL,  # entity_local_index (cell integral)
                ffi.NULL,  # cell_orientation
            )
            cell_matrix = np.linalg.cholesky(cell_matrix)
            self._insert_in_block_diagonal_matrix(global_ind, cell_matrix, block_diagonal_matrix)
            self._insert_in_local_global_dof_matrix(
                global_ind, global_cell_dofs, local_global_dof_matrix
            )
