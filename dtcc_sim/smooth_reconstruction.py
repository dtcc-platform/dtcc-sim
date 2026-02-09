"""
Smooth Field Reconstruction from Point Observations

A PDE-based method for reconstructing smooth scalar fields from sparse sensor measurements.
Uses Tikhonov regularization with point observation penalties to create continuous fields
from discrete sensor readings (e.g., air quality measurements).

The reconstruction minimizes:
    E(u) = ½w Σᵢ(u(xᵢ) - yᵢ)² + ½λ∫|∇u|²dx + ½α∫(u - u_bg)²dx

where:
- u is the reconstructed field
- xᵢ, yᵢ are sensor locations and measurements
- w is the data fidelity weight
- λ controls smoothness (gradient penalty)
- α anchors the field to a background value u_bg

This produces a linear system: A u = b with
    A = w(PᵀP) + λK + αM
    b = w(Pᵀy) + αM u_bg

where K is the stiffness matrix, M is the mass matrix, and P interpolates from
mesh dofs to sensor points using barycentric coordinates.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple
from pathlib import Path

import numpy as np
import dolfinx
from pydantic import BaseModel, Field

from dtcc_sim.fenics import *  # FEniCSx wrapper


# -----------------------------------------------------------------------------
# PETSc solver options (same as UrbanHeat)
# -----------------------------------------------------------------------------

DEFAULT_PETSC_OPTIONS: Mapping[str, Any] = {
    "ksp_monitor_short": None,
    "ksp_converged_reason": None,
    "ksp_type": "cg",
    "ksp_rtol": 1.0e-6,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
}


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------


class SmoothReconstructionParameters(BaseModel):
    """Parameters for smooth field reconstruction from point observations.

    This uses Pydantic BaseModel for validation and compatibility with
    the dtcc-core dataset system.
    """

    # Function space
    degree: int = Field(1, description="Polynomial degree (currently only 1 supported)")

    # Reconstruction weights
    lambda_smooth: float = Field(
        1.0, description="Smoothness regularization weight (gradient penalty)", gt=0
    )
    alpha: float = Field(
        1e-3, description="Background anchoring weight (mass penalty)", ge=0
    )
    data_weight: float = Field(
        100.0, description="Data fidelity weight (point observations)", gt=0
    )

    # Background
    background_value: Optional[float] = Field(
        None, description="Background field value (None = use mean of observations)"
    )

    # Volume mesh parameters (for when built from bounds)
    mesh_max_mesh_size: float = Field(25.0, description="Maximum mesh size in meters")
    mesh_domain_height: float = Field(80.0, description="Domain height in meters")
    mesh_raster_cell_size: float = Field(2.0, description="Terrain raster cell size")
    mesh_raster_radius: float = Field(
        3.0, description="Terrain raster interpolation radius"
    )

    # Robustness options
    z_offset: float = Field(
        0.0, description="Vertical offset to add to point z-coordinates"
    )
    point_search_tol: float = Field(
        1e-3, description="Tolerance for point location / barycentric coords"
    )


# -----------------------------------------------------------------------------
# Smooth Reconstruction Simulator
# -----------------------------------------------------------------------------


class SmoothReconstructionSimulator:
    """Reconstructs smooth fields from sparse point measurements using PDE smoothing.

    This is a generic simulator that takes point observations and creates a smooth
    continuous field using Tikhonov regularization. It does not depend on any specific
    data source (e.g., air quality) - that logic should be handled by the caller/dataset.

    Example:
        >>> # Point observations (Nx3 coordinates, N values)
        >>> points = np.array([[x1, y1, z1], [x2, y2, z2], ...])
        >>> values = np.array([v1, v2, ...])
        >>>
        >>> sim = SmoothReconstructionSimulator(
        ...     bounds=(xmin, ymin, xmax, ymax),
        ...     point_coords=points,
        ...     point_values=values,
        ...     params=SmoothReconstructionParameters(
        ...         lambda_smooth=1.0,
        ...         alpha=1e-3,
        ...         data_weight=100.0
        ...     )
        ... )
        >>> volume_mesh = sim.simulate()
        >>> # volume_mesh is a dtcc-core VolumeMesh with a Field attached
    """

    def __init__(
        self,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        mesh_path: Optional[str] = None,
        mesh: Optional[dolfinx.mesh.Mesh] = None,
        point_coords: Optional[np.ndarray] = None,
        point_values: Optional[np.ndarray] = None,
        field_name: str = "reconstructed_field",
        field_unit: str = "",
        params: Optional[SmoothReconstructionParameters] = None,
        petsc_options: Optional[Mapping[str, Any]] = None,
    ):
        """Initialize the smooth reconstruction simulator.

        Args:
            bounds: Bounding box (xmin, ymin, xmax, ymax) in CRS units
            mesh_path: Path to existing mesh file (.xdmf)
            mesh: Pre-loaded dolfinx mesh
            point_coords: Nx3 array of observation point coordinates
            point_values: N array of observation values
            field_name: Name for the reconstructed field (e.g., "NO2", "temperature")
            field_unit: Unit for the field (e.g., "µg/m³", "°C")
            params: Reconstruction parameters
            petsc_options: PETSc solver options (default: CG + boomeramg)
        """
        self.bounds = bounds
        self.mesh_path = mesh_path
        self.mesh = mesh
        self.point_coords = point_coords
        self.point_values = point_values
        self.field_name = field_name
        self.field_unit = field_unit
        self.params = params or SmoothReconstructionParameters()
        self.petsc_options = (
            petsc_options if petsc_options is not None else dict(DEFAULT_PETSC_OPTIONS)
        )

        # Will be populated during simulation
        self.volume_mesh_dtcc: Optional[Any] = None  # dtcc-core VolumeMesh
        self.solution: Optional[Function] = None

        # Enforce P1
        if self.params.degree != 1:
            raise ValueError("Only degree=1 (P1) is currently supported")

    def _build_mesh_from_bounds(self) -> None:
        """Build volume mesh from bounds using dtcc_core.datasets.city_volume_mesh."""
        info("SmoothReconstruction: Building volume mesh from bounds...")

        try:
            import dtcc_core.datasets as datasets
        except ImportError:
            raise ImportError(
                "dtcc_core is required to build mesh from bounds. "
                "Install it or provide mesh_path/mesh directly."
            )

        # Build volume mesh using the dataset
        volume_mesh = datasets.city_volume_mesh(
            bounds=self.bounds,
            max_mesh_size=self.params.mesh_max_mesh_size,
            domain_height=self.params.mesh_domain_height,
            raster_cell_size=self.params.mesh_raster_cell_size,
            raster_radius=self.params.mesh_raster_radius,
        )

        # Store for later use in output
        self.volume_mesh_dtcc = volume_mesh

        # Save to temporary file and load with FEniCSx
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".xdmf", delete=False) as tmp:
            tmp_path = tmp.name

        info(f"SmoothReconstruction: Saving mesh to temporary file: {tmp_path}")
        volume_mesh.save(tmp_path)

        # Load with FEniCSx (no markers needed for reconstruction)
        self.mesh = load_mesh(tmp_path)

        # Clean up temporary files
        Path(tmp_path).unlink(missing_ok=True)
        h5_path = Path(tmp_path).with_suffix(".h5")
        h5_path.unlink(missing_ok=True)

    def _load_if_needed(self) -> None:
        """Load or build mesh if not already available."""
        if self.mesh is not None:
            return

        if self.bounds is not None:
            self._build_mesh_from_bounds()
        elif self.mesh_path is not None:
            info(f"SmoothReconstruction: Loading mesh from {self.mesh_path}")
            self.mesh = load_mesh(self.mesh_path)
        else:
            raise ValueError(
                "SmoothReconstructionSimulator: provide bounds=..., mesh_path=..., or mesh=..."
            )

    def _prepare_point_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and validate point observation data.

        Returns:
            Tuple of (point_coords, point_values) with NaNs filtered and z_offset applied
        """
        if self.point_coords is None or self.point_values is None:
            raise ValueError(
                "SmoothReconstructionSimulator requires point_coords and point_values. "
                "These should be provided at initialization."
            )

        points = self.point_coords.copy()
        values = self.point_values.copy()

        # Filter out NaNs
        valid = ~np.isnan(values)
        points = points[valid]
        values = values[valid]

        # Apply z_offset
        if self.params.z_offset != 0:
            points[:, 2] += self.params.z_offset

        if len(values) == 0:
            raise ValueError(
                "No valid measurements found after filtering NaNs. "
                "Check input data quality."
            )

        info(f"SmoothReconstruction: Using {len(values)} point observations")

        return points, values

    def _locate_points(
        self, mesh: dolfinx.mesh.Mesh, points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Locate sensor points in the mesh.

        Args:
            mesh: dolfinx mesh
            points: Nx3 array of sensor coordinates

        Returns:
            cells: array of cell indices (or -1 if not found)
            mask: boolean array indicating which points were found
        """
        from dolfinx import geometry

        # Create bounding box tree
        bb_tree = geometry.bb_tree(mesh, mesh.geometry.dim)

        num_points = points.shape[0]
        cells = np.full(num_points, -1, dtype=np.int32)

        for i, point in enumerate(points):
            # Find candidate cells
            cell_candidates = geometry.compute_collisions_points(bb_tree, point)

            # For each candidate, check if point is actually inside using barycentric coords
            for cell_idx in cell_candidates.array:
                cell_vertices = mesh.geometry.dofmap[cell_idx]
                X = mesh.geometry.x[cell_vertices]

                # Check if point is inside this tet using barycentric coords
                phi = self._barycentric_weights_tet(
                    X, point, self.params.point_search_tol
                )
                if phi is not None:
                    cells[i] = cell_idx
                    break  # Found the cell, no need to check others

        mask = cells >= 0
        num_found = np.sum(mask)
        num_outside = num_points - num_found

        if num_outside > 0:
            info(
                f"SmoothReconstruction: {num_outside}/{num_points} sensor points outside mesh (skipped)"
            )

        return cells, mask

    def _barycentric_weights_tet(
        self, X: np.ndarray, x: np.ndarray, tol: float = 1e-8
    ) -> Optional[np.ndarray]:
        """Compute barycentric weights for a point in a tetrahedron.

        Args:
            X: (4, 3) array of tetrahedron vertex coordinates
            x: (3,) query point coordinates
            tol: tolerance for checking if point is inside

        Returns:
            (4,) array of barycentric weights, or None if point is outside
        """
        # Solve for w0, w1, w2 in: x = w0*X0 + w1*X1 + w2*X2 + w3*X3
        # where w3 = 1 - w0 - w1 - w2
        #
        # Rearrange: x = w0*X0 + w1*X1 + w2*X2 + (1-w0-w1-w2)*X3
        #            x = w0*(X0-X3) + w1*(X1-X3) + w2*(X2-X3) + X3
        #            x - X3 = [X0-X3, X1-X3, X2-X3] * [w0, w1, w2]^T

        A = np.column_stack([X[0] - X[3], X[1] - X[3], X[2] - X[3]])
        b = x - X[3]

        try:
            w_012 = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None  # Degenerate tet

        w3 = 1.0 - np.sum(w_012)
        weights = np.append(w_012, w3)

        # Check if point is inside (all weights should be >= 0 within tolerance)
        if np.any(weights < -tol):
            return None

        return weights

    def simulate(self, *, output_path: Optional[str] = None) -> Function:
        """Run the smooth field reconstruction.

        Args:
            output_path: Optional path to save solution (XDMF format)

        Returns:
            Reconstructed field as dolfinx Function
        """
        # Load mesh and prepare point data
        self._load_if_needed()
        point_coords, point_values = self._prepare_point_data()

        info("SmoothReconstruction: Setting up finite element problem...")

        # Create function space (P1 only)
        V = dolfinx.fem.functionspace(self.mesh, ("Lagrange", 1))

        # Define trial and test functions
        u = TrialFunction(V)
        v = TestFunction(V)

        # Define measures
        dx = Measure("dx", domain=self.mesh)

        # Assemble stiffness matrix K = ∫ ∇u·∇v dx
        info("SmoothReconstruction: Assembling stiffness matrix...")
        K_form = inner(grad(u), grad(v)) * dx
        K = assemble_matrix(K_form)
        K.assemble()

        # Assemble mass matrix M = ∫ u v dx
        info("SmoothReconstruction: Assembling mass matrix...")
        M_form = inner(u, v) * dx
        M = assemble_matrix(M_form)
        M.assemble()

        # Determine background value
        u_bg = (
            self.params.background_value
            if self.params.background_value is not None
            else float(np.mean(point_values))
        )
        info(f"SmoothReconstruction: Background value u_bg = {u_bg:.4f}")

        # Initialize A = λ*K + α*M
        A = K.copy()
        A.scale(self.params.lambda_smooth)
        A.axpy(self.params.alpha, M)
        A.assemble()

        # Initialize b = α * M * u_bg
        from petsc4py import PETSc

        u_bg_func = dolfinx.fem.Function(V)
        u_bg_func.x.array[:] = u_bg
        u_bg_func.x.scatter_forward()

        # Compute b = α * M * u_bg using matrix-vector product
        b = M.createVecRight()
        M.mult(u_bg_func.x.petsc_vec, b)
        b.scale(self.params.alpha)

        # Locate observation points in mesh
        info("SmoothReconstruction: Locating observation points in mesh...")
        cells, mask = self._locate_points(self.mesh, point_coords)

        valid_points = point_coords[mask]
        valid_values = point_values[mask]
        valid_cells = cells[mask]

        info(
            f"SmoothReconstruction: Adding {len(valid_values)} point observation penalties..."
        )

        # Add point observation penalties to A and b
        num_added = 0
        for i, (point, value, cell) in enumerate(
            zip(valid_points, valid_values, valid_cells)
        ):
            # Get cell vertex coordinates
            cell_vertices = self.mesh.geometry.dofmap[cell]
            X = self.mesh.geometry.x[cell_vertices]  # Shape: (4, 3) for tet

            # Compute barycentric weights
            phi = self._barycentric_weights_tet(X, point, self.params.point_search_tol)

            if phi is None:
                continue  # Point outside tet (shouldn't happen, but skip if it does)

            # Get dofs for this cell
            dofs = V.dofmap.list[cell]

            # Update A: A[dof_j, dof_k] += w * phi[j] * phi[k]
            w = self.params.data_weight
            for j in range(len(dofs)):
                for k in range(len(dofs)):
                    A.setValue(dofs[j], dofs[k], w * phi[j] * phi[k], addv=True)

            # Update b: b[dof_j] += w * value * phi[j]
            for j in range(len(dofs)):
                b.setValue(dofs[j], w * value * phi[j], addv=True)

            num_added += 1

        # Assemble A and b with accumulated values
        from petsc4py import PETSc

        A.assemble()
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        b.assemble()

        info(f"SmoothReconstruction: Successfully added {num_added} point constraints")

        # Solve the linear system
        info("SmoothReconstruction: Solving linear system...")
        from petsc4py import PETSc

        ksp = PETSc.KSP().create(self.mesh.comm)
        ksp.setOperators(A)
        ksp.setFromOptions()
        opts = PETSc.Options()
        for k, v in self.petsc_options.items():
            if v is None:
                opts[k] = None
            else:
                opts[k] = v
        ksp.setFromOptions()

        uh = dolfinx.fem.Function(V)
        ksp.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()

        info("SmoothReconstruction: Solution complete")

        # Store solution
        self.solution = uh

        # Output
        if output_path is not None:
            info(f"SmoothReconstruction: Saving solution to {output_path}")
            self.solution.save(output_path)

        return self.solution


__all__ = ["SmoothReconstructionParameters", "SmoothReconstructionSimulator"]
