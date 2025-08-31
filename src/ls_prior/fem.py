from numbers import Real
from typing import Annotated

import cffi
import dolfinx as dlx
import numpy as np
import ufl
from beartype.vale import Is
from dolfinx.fem import petsc
from mpi4py import MPI
from petsc4py import PETSc

ffi = cffi.FFI()


# ==================================================================================================
class FEMHandler:
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self, mesh: dlx.mesh.Mesh, fe_data: tuple[str, Annotated[int, Is[lambda x: x > 0]]]
    ) -> None:
        self.function_space = dlx.fem.functionspace(mesh, fe_data)

    # ----------------------------------------------------------------------------------------------
    def generate_forms(
        self,
        kappa: Annotated[Real, Is[lambda x: x > 0]],
        tau: Annotated[Real, Is[lambda x: x > 0]],
        robin_const: float | None = None,
    ) -> tuple[ufl.Form, ufl.Form]:
        trial_function = ufl.TrialFunction(self.function_space)
        test_function = ufl.TestFunction(self.function_space)
        mass_matrix_form = ufl.inner(trial_function, test_function) * ufl.dx
        stiffness_matrix_form = (
            ufl.inner(ufl.grad(trial_function), ufl.grad(test_function)) * ufl.dx
        )
        spde_matrix_form = kappa**2 * tau * mass_matrix_form + tau * stiffness_matrix_form
        if robin_const is not None:
            robin_boundary_form = robin_const * ufl.inner(trial_function, test_function) * ufl.ds
            spde_matrix_form += robin_boundary_form
        return mass_matrix_form, spde_matrix_form

    # ----------------------------------------------------------------------------------------------
    def assemble_matrix(self, form: ufl.Form) -> PETSc.Mat:
        matrix = petsc.assemble_matrix(dlx.fem.form(form))
        matrix.assemble()
        return matrix


# ==================================================================================================
class FEMMatrixBlockFactorization:
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self, mesh: dlx.mesh.Mesh, function_space: dlx.fem.FunctionSpace, form: ufl.Form
    ) -> None:
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
        block_diagonal_matrix, local_global_dof_matrix = self._set_up_petsc_mats()
        self._assemble_matrices_over_cells(block_diagonal_matrix, local_global_dof_matrix)
        local_global_dof_matrix.transpose()
        matrix_factorization = local_global_dof_matrix.matMult(block_diagonal_matrix)
        return matrix_factorization

    # ----------------------------------------------------------------------------------------------
    def _init_assembly_kernel(self, mpi_communicator: MPI.Comm, form: ufl.Form) -> None:
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
        global_ind: int,
        cell_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        block_diagonal_matrix: PETSc.Mat,
    ) -> None:
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
        global_ind: int,
        global_cell_dofs: np.ndarray[tuple[int], np.dtype[np.float64]],
        local_global_dof_matrix: PETSc.Mat,
    ) -> None:
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

        block_diagonal_matrix.assemble()
        local_global_dof_matrix.assemble()
