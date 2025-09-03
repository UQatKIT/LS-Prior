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
    r"""Construct dolfinx forms for the mass matrix and SPDE system matrix.

    This method constructs dolfinx variational forms that resemble the mass matrix contribution
    and SPDE system matrix, i.e. the left-hand-side of the SPDE generating the random field.
    More specifically, let $\Omega$ be the domain of interest, and $\phi$ both the trial and test
    function (defined over the same function space).
    The mass matrix contribution is then given as $(\phi, \phi)_{L^2(\Omega)}$, and the total SPDE
    matrix contribution is

    $$
    \begin{equation*}
        kappa^2 tau (\phi, \phi)_{L^2(\Omega)} + tau (\nabla \phi, \nabla \phi)_{L^2(\Omega)}
        + \beta * (\phi, \phi)_{L^2(\Omega)}
    \end{equation*}
    $$

    $\kappa$ and $\tau$ correspond to the parameters `kappa` and `tau` for the parameterization of
    the field.
    $\beta$ is the optional `robin_const` parameter that enforces Robin boundary
    conditions of the form $\nabla \phi \cdot n + \beta \phi = 0$, instead of homogeneous Neumann
    boundary conditions.

    Args:
        function_space (dlx.fem.FunctionSpace): dolfinx function space to construct forms over.
        kappa (Real): Parameter $\kappa$ of the prior field.
        tau (Real): Parameter $\tau$ of the prior field.
        robin_const (Real | None, optional): Parameter $\beta$ of the prior field enforcing
            Robin boundary conditions. Defaults to None.

    Returns:
        tuple[ufl.Form, ufl.Form]: dolfinx forms for the mass matrix and SPDE system matrix
            contributions.
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
    """Converter between vertex based data and DoF representation on a dolfinx function space.

    This class connects the representation of arrays in dolfinx on a specified function space
    with a vertex-based viewpoint. Think of it as the adapter required for the prior to
    communicate with external components. The underlying idea is that such outside components
    only see the computational mesh, and define discrete data over the vertices of that mesh
    A `FEMConverter` object takes such data structures and inerpolates them to the provided
    function space. On the other hand, it can interpolate any data defined on the DoFs of the
    underlying function space onto the vertices of the mesh.
    Internally, the `FEMConverter` assigns vertex based input data to the DoFs of a P1
    function space (whose degrees of freedom are exactly the vertices).
    It subsequently utilizes dolfinx's efficient interpolation between function spaces.
    On the other hand, data from some function space is interpolated to a P1 space, and
    subsequently extracts vertex values.

    Methods:
        convert_vertex_values_to_dofs: Convert vertex based data to DoF representation
        convert_dofs_to_vertex_values: Convert DoF based data to vertex representation
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, function_space: dlx.fem.FunctionSpace) -> None:
        """Initialize the converter for a given dolfinx function space.

        The function space implicitly carries the mesh, and thus the vertices we want to convert
        from/to.

        Args:
            function_space (dlx.fem.FunctionSpace): Function space in which degrees of freedom lie.
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
        """Convert vertex based data to DoF representation.

        Args:
            vertex_values (np.ndarray[tuple[int], np.dtype[np.float64]]): Array of data defined on
            the vertices of the underlying computational mesh.

        Raises:
            ValueError: Checks that `vertex_values` has the same dimension as a P1 function space
                on the underlying mesh.

        Returns:
            PETSc.Vec: PETSc Vector containing the input data, interpolated to the required
                function space DoFs.
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
        """Convert DoF based data to vertex representation.

        Args:
            dof_values (PETSc.Vec): PETSc Vector containing Data on the DoFs of the underlying
                function space.

        Raises:
            ValueError: Checks that the dimension of the input vector matches the number of DoFs
                on the underlying function space.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Array of data interpolated to the vertices
                of the underlying mesh.
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
class FEMMatrixFactorizationAssembler:
    r"""Assembler for the rectangular factorization of the FEM matrix.

    The rectangular FEM matrix factorization we assemble is of the form $BP$, where
    $B\in\mathbb{R}^{MN_{\text{elem}} \times MN_{\text{elem}}}$ is a
    block-diagonal matrix whose block consist of the local FEM matrix Cholesky factor of each
    mesh cell, and $P\in\mathbb{R}^{MN_{\text{elem}} \times N_{\text{DoF}}}$ contains the
    mapping from each row of $B$ to the global DoFs of the matrix entries resembled by that row.

    !!! warning
        The assembly procedure uses internals of dolfinx, which might be subject to change in
        future versions.

    Methods:
        assemble: Assemble the block factorization.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self, mesh: dlx.mesh.Mesh, function_space: dlx.fem.FunctionSpace, form: ufl.Form
    ) -> None:
        """Initialize the block factorization assembler.

        Args:
            mesh (dlx.mesh.Mesh): Dolfinx Mesh object representing the computational domain.
            function_space (dlx.fem.FunctionSpace): Function space of the FEM problem.
            form (ufl.Form): Weak form of the FEM matrix to factorize.
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
    def assemble(self) -> tuple[PETSc.Mat, PETSc.Mat]:
        r"""Assemble the rectangular matrix factorization.

        Returns:
            tuple[PETSc.Mat, PETSc.Mat]: PETSc Matrices $B$ and $P$ representing the
                rectangular factorization of an FEM matrix.
        """
        block_diagonal_matrix, dof_map_matrix = self._set_up_petsc_mats()
        self._assemble_matrices_over_cells(block_diagonal_matrix, dof_map_matrix)
        block_diagonal_matrix.assemble()
        dof_map_matrix.assemble()
        return block_diagonal_matrix, dof_map_matrix

    # ----------------------------------------------------------------------------------------------
    def _init_assembly_kernel(self, mpi_communicator: MPI.Comm, form: ufl.Form) -> cffi.FFI.CData:
        """Initialize the dolfinx assembly kernel.

        Kernel objects in dolfinx generate local matrix contributions over a single mesh cell,
        corresponding to a provided weak form. They are typically called internally during
        the assembly process of FEM system matrices.
        Under the hood, this method uses the `ffcx` jit compiler, and returns the resulting kernel's
        `tabulate_tensor` method.

        !!! warning
            This method uses internals of dolfinx, which might be subject to change in future
            versions.

        Args:
            mpi_communicator (MPI.Comm): MPI communicator for the kernel to use.
            form (ufl.Form): Weak form to compile per cell.

        Returns:
            cffi.FFI.CData: Compiled callable, returning the local FEM matrix contribution for a
                a single mesh cell.
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
        r"""Initialize the PETSc matrices $B$ and $P$for the assembly process.

        Returns:
            tuple[PETSc.Mat, PETSc.Mat]: Block diagonal matrix $B$ and local-to-global
                DoF matrix $P$.
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

        dof_map_matrix = PETSc.Mat().createAIJ(
            [self._num_global_cells * self._num_cell_dofs, self._num_global_dofs],
            comm=self._mpi_communicator,
        )
        dof_map_matrix.setPreallocationNNZ(1)
        dof_map_matrix.setUp()

        return block_diagonal_matrix, dof_map_matrix

    # ----------------------------------------------------------------------------------------------
    def _insert_in_block_diagonal_matrix(
        self,
        global_ind: np.integer,
        cell_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        block_diagonal_matrix: PETSc.Mat,
    ) -> None:
        r"""Insert FEM matrix Cholesky factor of single mesh cell into block-diagonal matrix $B$.

        Insertion is done in-place.

        Args:
            global_ind (np.integer): Global index of the current mesh cell.
            cell_matrix (np.ndarray[tuple[int, int], np.dtype[np.float64]]): Local cell matrix
                contribution values.
            block_diagonal_matrix (PETSc.Mat): Global matrix $B$ to insert into.
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
    def _insert_in_dof_map_matrix(
        self,
        global_ind: np.integer,
        global_cell_dofs: np.ndarray[tuple[int], np.dtype[np.integer]],
        dof_map_matrix: PETSc.Mat,
    ) -> None:
        """Insert global DoFs of a single mesh cell into local-to-global DoF matrix $P$.

        Insertion is done in-place.

        Args:
            global_ind (np.integer): Global index of the current mesh cell.
            global_cell_dofs (np.ndarray[tuple[int], np.dtype[np.integer]]):
                Global indices of the DoFs in the current cell.
            dof_map_matrix (PETSc.Mat): Local-to-global DoF matrix $P$.
        """
        row_inds = np.arange(
            global_ind * self._num_cell_dofs,
            (global_ind + 1) * self._num_cell_dofs,
            dtype=PETSc.IntType,
        )
        for row_ind, col_ind in zip(row_inds, global_cell_dofs, strict=True):
            dof_map_matrix.setValues(
                row_ind,
                col_ind,
                1.0,
                addv=PETSc.InsertMode.INSERT_VALUES,
            )

    # ----------------------------------------------------------------------------------------------
    def _assemble_matrices_over_cells(
        self, block_diagonal_matrix: PETSc.Mat, dof_map_matrix: PETSc.Mat
    ) -> None:
        """Assemble the matrices $B$ and $P$.

        The method loops over all cells and invokes the kernel for the local FEM matrix
        contributions. It further computes the Cholesky factor of each local matrix, and
        inserts the result and global index mapping into the matrices $B$ and $P$, respectively.

        Args:
            block_diagonal_matrix (PETSc.Mat): Block diagonal matrix $B$ for cell Cholesky factors.
            dof_map_matrix (PETSc.Mat): Local-to-global DoF matrix $P$.
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
            self._insert_in_dof_map_matrix(global_ind, global_cell_dofs, dof_map_matrix)
