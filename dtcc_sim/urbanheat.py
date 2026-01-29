"""
Urban Heat Simulator

A steady-state diffusion model for air temperature in a 3D city volume.
Supports multiple boundary condition types (Dirichlet, Neumann, Robin)
for different urban surfaces (walls, roofs, ground, open boundaries).
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Mapping, Optional, Union
from pathlib import Path

import numpy as np
import dolfinx
from pydantic import BaseModel, Field

from dtcc_sim.fenics import *  # FEniCSx wrapper


# -----------------------------------------------------------------------------
# Boundary categories (collapsed markers)
# -----------------------------------------------------------------------------


class BndCat(IntEnum):
    """Boundary category enumeration for urban surfaces."""

    WALL = 1
    ROOF = 2
    GROUND = 3
    OPEN = 4


# -----------------------------------------------------------------------------
# Boundary condition specifications
# -----------------------------------------------------------------------------


class DirichletBCSpec(BaseModel):
    """Dirichlet boundary condition: T = value"""

    type: str = "dirichlet"
    value: float = Field(..., description="Temperature value")


class NeumannBCSpec(BaseModel):
    """Neumann boundary condition: -k dT/dn = flux"""

    type: str = "neumann"
    flux: float = Field(..., description="Heat flux (outward normal)")


class RobinBCSpec(BaseModel):
    """Robin boundary condition: -k dT/dn = h(T - value)"""

    type: str = "robin"
    h: float = Field(..., description="Heat transfer coefficient")
    value: float = Field(..., description="Reference temperature")


BCSpec = Union[DirichletBCSpec, NeumannBCSpec, RobinBCSpec]


# -----------------------------------------------------------------------------
# Simulation parameters
# -----------------------------------------------------------------------------


class UrbanHeatParameters(BaseModel):
    """Parameters for urban heat simulation.

    This uses Pydantic BaseModel for validation and JSON schema generation,
    making it compatible with the dtcc-core dataset system.
    """

    # PDE coefficients
    kappa: float = Field(1.0, description="Thermal diffusivity coefficient", gt=0)
    sigma: float = Field(
        0.0,
        description="Relaxation/reaction coefficient (for sigma*(T - T_ambient) term)",
        ge=0,
    )
    T_ambient: float = Field(
        0.0,
        description="Ambient temperature (reference for relaxation and open boundaries)",
    )

    # Function space
    degree: int = Field(
        1, description="Polynomial degree for finite element space", ge=1, le=3
    )

    # Boundary conditions (using dict representation for Pydantic compatibility)
    # Default: hot buildings, ambient open boundary (anomaly form)
    wall_bc_type: str = Field(
        "robin", description="BC type for walls: dirichlet, neumann, or robin"
    )
    wall_value: float = Field(
        1.0, description="Temperature/reference value for wall BC"
    )
    wall_h: float = Field(
        5.0, description="Heat transfer coeff for wall (Robin BC only)"
    )
    wall_flux: float = Field(0.0, description="Heat flux for wall (Neumann BC only)")

    roof_bc_type: str = Field("robin", description="BC type for roofs")
    roof_value: float = Field(
        1.0, description="Temperature/reference value for roof BC"
    )
    roof_h: float = Field(
        5.0, description="Heat transfer coeff for roof (Robin BC only)"
    )
    roof_flux: float = Field(0.0, description="Heat flux for roof (Neumann BC only)")

    ground_bc_type: str = Field("robin", description="BC type for ground")
    ground_value: float = Field(
        0.0, description="Temperature/reference value for ground BC"
    )
    ground_h: float = Field(
        2.0, description="Heat transfer coeff for ground (Robin BC only)"
    )
    ground_flux: float = Field(
        0.0, description="Heat flux for ground (Neumann BC only)"
    )

    open_bc_type: str = Field("dirichlet", description="BC type for open boundaries")
    open_value: float = Field(
        0.0, description="Temperature/reference value for open BC"
    )
    open_h: float = Field(
        1.0, description="Heat transfer coeff for open (Robin BC only)"
    )
    open_flux: float = Field(0.0, description="Heat flux for open (Neumann BC only)")

    # Volume mesh parameters (for when built from bounds)
    mesh_max_mesh_size: float = Field(25.0, description="Maximum mesh size in meters")
    mesh_domain_height: float = Field(80.0, description="Domain height in meters")
    mesh_raster_cell_size: float = Field(2.0, description="Terrain raster cell size")
    mesh_raster_radius: float = Field(
        3.0, description="Terrain raster interpolation radius"
    )

    def get_bc_spec(self, surface: str) -> BCSpec:
        """Get boundary condition specification for a surface type.

        Args:
            surface: One of 'wall', 'roof', 'ground', 'open'

        Returns:
            BCSpec object (Dirichlet, Neumann, or Robin)
        """
        bc_type = getattr(self, f"{surface}_bc_type")
        value = getattr(self, f"{surface}_value")
        h = getattr(self, f"{surface}_h")
        flux = getattr(self, f"{surface}_flux")

        if bc_type == "dirichlet":
            return DirichletBCSpec(value=value)
        elif bc_type == "neumann":
            return NeumannBCSpec(flux=flux)
        elif bc_type == "robin":
            return RobinBCSpec(h=h, value=value)
        else:
            raise ValueError(f"Unknown BC type: {bc_type}")


DEFAULT_PETSC_OPTIONS: Mapping[str, Any] = {
    "ksp_monitor_short": None,
    "ksp_converged_reason": None,
    "ksp_type": "cg",
    "ksp_rtol": 1.0e-6,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
}


# -----------------------------------------------------------------------------
# Helpers: global marker stats + collapsing to 4 categories
# -----------------------------------------------------------------------------


def _global_max_positive_marker(
    mesh: dolfinx.mesh.Mesh, markers: dolfinx.mesh.MeshTags
) -> int:
    """Find the global maximum positive marker value across all processes."""
    vals = np.asarray(markers.values, dtype=np.int32)
    if np.any(vals >= 0):
        local_max = int(vals[vals >= 0].max())
    else:
        local_max = -1
    return int(mesh.comm.allreduce(local_max, op=MPI.MAX))


def infer_num_buildings(mesh: dolfinx.mesh.Mesh, markers: dolfinx.mesh.MeshTags) -> int:
    """
    Infer the number of buildings from marker scheme.

    Assumes marker scheme:
      walls: 0..N-1
      roofs: N..2N-1
      other boundaries: negative tags

    Args:
        mesh: The mesh
        markers: Facet markers

    Returns:
        Number of buildings
    """
    max_pos = _global_max_positive_marker(mesh, markers)
    if max_pos < 0:
        return 0
    return (max_pos + 1) // 2


def collapse_boundary_markers(
    mesh: dolfinx.mesh.Mesh,
    markers: dolfinx.mesh.MeshTags,
    num_buildings: int,
    ground_tag: int = -1,
) -> dolfinx.mesh.MeshTags:
    """
    Create collapsed facet MeshTags with 4 categories: WALL/ROOF/GROUND/OPEN.

    This reduces O(num_buildings) subdomain terms in UFL forms to O(1).

    Args:
        mesh: The mesh
        markers: Original facet markers
        num_buildings: Number of buildings
        ground_tag: Marker value for ground facets (default: -1)

    Returns:
        New MeshTags with collapsed categories
    """
    fdim = mesh.topology.dim - 1

    idx = np.asarray(markers.indices, dtype=np.int32)
    vals = np.asarray(markers.values, dtype=np.int32)

    N = int(num_buildings)
    cat = np.empty_like(vals, dtype=np.int32)

    # Categorize facets
    mask_wall = (vals >= 0) & (vals < N)
    mask_roof = (vals >= N) & (vals < 2 * N)
    mask_ground = vals == ground_tag
    mask_open = vals < ground_tag  # typically -2,-3,... (bbox sides/top)

    cat[mask_wall] = int(BndCat.WALL)
    cat[mask_roof] = int(BndCat.ROOF)
    cat[mask_ground] = int(BndCat.GROUND)
    cat[mask_open] = int(BndCat.OPEN)

    # Anything unexpected: treat as OPEN (safe default)
    mask_other = ~(mask_wall | mask_roof | mask_ground | mask_open)
    cat[mask_other] = int(BndCat.OPEN)

    # meshtags expects sorted indices for robust .find()
    order = np.argsort(idx)
    return dolfinx.mesh.meshtags(mesh, fdim, idx[order], cat[order])


# -----------------------------------------------------------------------------
# Simulator
# -----------------------------------------------------------------------------


class UrbanHeatSimulator:
    """
    Urban Heat Simulator: Steady-state thermal analysis for 3D urban environments.

    Physical Model
    --------------
    This simulator solves the steady-state heat diffusion equation with optional
    relaxation term to model air temperature distribution in urban volumes:

        -∇·(κ∇T) + σ(T - T_ambient) = 0    in Ω (urban domain)

    where:
        - T: temperature field [°C or K]
        - κ: thermal diffusivity [m²/s] - models heat conduction/diffusion in air
        - σ: relaxation coefficient [1/s] - models thermal exchange with ambient air mass
        - T_ambient: ambient/background temperature [°C or K]

    The diffusion term (-∇·(κ∇T)) represents heat conduction through the air, while
    the relaxation term (σ(T - T_ambient)) models heat exchange with the surrounding
    ambient air mass (e.g., through large-scale atmospheric mixing, advection effects
    simplified as a sink/source term).

    Boundary Conditions
    -------------------
    Three types of boundary conditions are supported on urban surfaces:

    1. **Dirichlet BC**: Fixed temperature
       T = T_prescribed    on ∂Ω

    2. **Neumann BC**: Prescribed heat flux
       -κ ∂T/∂n = q    on ∂Ω

    3. **Robin BC**: Convective heat transfer (Newton's law of cooling)
       -κ ∂T/∂n = h(T - T_ref)    on ∂Ω

       where h is the heat transfer coefficient [W/(m²·K)] and T_ref is the
       reference temperature (e.g., building surface temperature).

    Urban surfaces are categorized as:
        - **Walls**: Building vertical surfaces
        - **Roofs**: Building horizontal surfaces (top)
        - **Ground**: Terrain surface
        - **Open**: Far-field/domain boundaries

    Applications
    ------------
    - **Urban Heat Island (UHI) analysis**: Quantify temperature elevation in cities
    - **Building energy assessment**: Evaluate heat losses and thermal comfort
    - **Outdoor thermal comfort**: Study pedestrian-level temperature distribution
    - **Urban climate adaptation**: Inform urban planning and design decisions
    - **Mitigation strategies**: Evaluate green infrastructure, cool roofs, ventilation

    Physical Interpretation
    -----------------------
    Buildings act as heat sources/sinks with surface temperatures typically different
    from ambient air. The Robin boundary condition models convective heat exchange
    between building surfaces and surrounding air. The steady-state solution represents
    the equilibrium temperature distribution under given thermal conditions.

    Typical parameter ranges:
        - κ ~ 0.5-5.0 m²/s (effective thermal diffusivity of urban air)
        - σ ~ 0.0-0.5 1/s (relaxation rate, 0 = pure diffusion)
        - h ~ 2-10 W/(m²·K) for natural convection
        - h ~ 10-50 W/(m²·K) for forced convection (wind)

    Usage Examples
    --------------
    >>> # From bounds (auto-generates mesh)
    >>> sim = UrbanHeatSimulator(bounds=bounds)
    >>> T = sim.simulate()

    >>> # From pre-computed mesh
    >>> sim = UrbanHeatSimulator(mesh_path="mesh.xdmf")
    >>> T = sim.simulate()

    >>> # Heat wave scenario with custom parameters
    >>> params = UrbanHeatParameters(
    ...     kappa=2.0,           # Enhanced mixing
    ...     T_ambient=20.0,      # 20°C ambient
    ...     wall_value=35.0,     # Hot building surfaces
    ...     wall_h=8.0,          # Moderate convection
    ... )
    >>> sim = UrbanHeatSimulator(bounds=bounds, params=params)
    >>> T = sim.simulate(output_path="heatwave.xdmf")

    Notes
    -----
    - The simulator uses FEniCSx for finite element discretization
    - Linear (P1) or quadratic (P2) Lagrange elements supported
    - Parallel computation via MPI (inherited from FEniCSx)
    - Boundary markers are automatically collapsed from per-building to
      per-category to reduce computational cost

    See Also
    --------
    UrbanHeatParameters : Parameter specification and validation
    dtcc_sim.datasets.UrbanHeatSimulationDataset : Dataset wrapper interface
    """

    def __init__(
        self,
        *,
        bounds: Optional[Any] = None,
        mesh_path: Optional[str] = None,
        mesh: Optional[dolfinx.mesh.Mesh] = None,
        markers: Optional[dolfinx.mesh.MeshTags] = None,
        params: Optional[UrbanHeatParameters] = None,
        petsc_options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Initialize the urban heat simulator.

        Args:
            bounds: DTCC Bounds object or bounds tuple [minx, miny, maxx, maxy]
            mesh_path: Path to XDMF mesh file with markers
            mesh: Pre-loaded dolfinx mesh
            markers: Pre-loaded facet markers
            params: Simulation parameters
            petsc_options: PETSc solver options

        Note: Provide either bounds, mesh_path, or (mesh, markers).
        """
        self.bounds = bounds
        self.mesh_path = mesh_path
        self.mesh = mesh
        self.markers = markers

        self.params = params if params is not None else UrbanHeatParameters()
        self.petsc_options = (
            dict(petsc_options)
            if petsc_options is not None
            else dict(DEFAULT_PETSC_OPTIONS)
        )

        self.num_buildings: Optional[int] = None
        self.category_markers: Optional[dolfinx.mesh.MeshTags] = None
        self.solution: Optional[Function] = None

    def _build_mesh_from_bounds(self) -> None:
        """Build volume mesh from bounds using dtcc_core.datasets.volumemesh."""
        info("UrbanHeat: Building volume mesh from bounds...")

        try:
            import dtcc_core.datasets as datasets
        except ImportError:
            raise ImportError(
                "dtcc_core is required to build mesh from bounds. "
                "Install it or provide mesh_path/mesh directly."
            )

        # Build volume mesh using the dataset
        volume_mesh = datasets.volumemesh(
            bounds=self.bounds,
            max_mesh_size=self.params.mesh_max_mesh_size,
            domain_height=self.params.mesh_domain_height,
            raster_cell_size=self.params.mesh_raster_cell_size,
            raster_radius=self.params.mesh_raster_radius,
        )

        # Save to temporary file and load with FEniCSx
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".xdmf", delete=False) as tmp:
            tmp_path = tmp.name

        info(f"UrbanHeat: Saving mesh to temporary file: {tmp_path}")
        volume_mesh.save(tmp_path)

        # Load with FEniCSx
        self.mesh, self.markers = load_mesh_with_markers(tmp_path)

        # Clean up temporary file
        Path(tmp_path).unlink(missing_ok=True)
        h5_path = Path(tmp_path).with_suffix(".h5")
        h5_path.unlink(missing_ok=True)

    def _load_if_needed(self) -> None:
        """Load or build mesh if not already available."""
        if self.mesh is not None and self.markers is not None:
            return

        if self.bounds is not None:
            self._build_mesh_from_bounds()
        elif self.mesh_path is not None:
            info(f"UrbanHeat: Loading mesh from {self.mesh_path}")
            self.mesh, self.markers = load_mesh_with_markers(self.mesh_path)
        else:
            raise ValueError(
                "UrbanHeatSimulator: provide bounds=..., mesh_path=..., or (mesh=..., markers=...)."
            )

    def _apply_bc_spec(
        self,
        V: dolfinx.fem.FunctionSpace,
        u: Any,
        v: Any,
        ds_cat: Any,
        cat_markers: dolfinx.mesh.MeshTags,
        spec: BCSpec,
        category: BndCat,
        a: Any,
        L: Any,
        bcs: list,
    ) -> tuple[Any, Any, list]:
        """Apply a boundary condition specification to the variational forms.

        Args:
            V: Function space
            u: Trial function
            v: Test function
            ds_cat: Measure over boundary categories
            cat_markers: Collapsed category markers
            spec: Boundary condition specification
            category: Boundary category
            a: Bilinear form
            L: Linear form
            bcs: List of Dirichlet BCs

        Returns:
            Updated (a, L, bcs) tuple
        """
        tag = int(category)

        if isinstance(spec, DirichletBCSpec):
            bcs.append(
                DirichletBC(V, spec.value, markers=cat_markers, marker_value=tag)
            )
            return a, L, bcs

        if isinstance(spec, NeumannBCSpec):
            q = Constant(self.mesh, float(spec.flux))
            L = L + q * v * ds_cat(tag)
            return a, L, bcs

        if isinstance(spec, RobinBCSpec):
            h = Constant(self.mesh, float(spec.h))
            Tref = Constant(self.mesh, float(spec.value))
            a = a + h * u * v * ds_cat(tag)
            L = L + h * Tref * v * ds_cat(tag)
            return a, L, bcs

        raise TypeError(f"Unsupported BC spec type: {type(spec)}")

    def simulate(self, *, output_path: Optional[str] = None) -> Function:
        """Run the urban heat simulation.

        Args:
            output_path: Optional path to save solution (XDMF format)

        Returns:
            Temperature field as dolfinx Function
        """
        self._load_if_needed()
        assert self.mesh is not None and self.markers is not None

        # Infer buildings + collapse tags
        self.num_buildings = infer_num_buildings(self.mesh, self.markers)
        info(f"UrbanHeat: inferred num_buildings = {self.num_buildings}")

        self.category_markers = collapse_boundary_markers(
            self.mesh, self.markers, self.num_buildings
        )

        # Function space
        V = FunctionSpace(self.mesh, "Lagrange", self.params.degree)
        info(
            f"UrbanHeat: Function space has {V.dofmap.index_map.size_global} degrees of freedom"
        )

        # Measures over collapsed boundary categories
        ds_cat = Measure("ds", domain=self.mesh, subdomain_data=self.category_markers)

        # Unknown/test functions
        u = TrialFunction(V)
        v = TestFunction(V)

        # Coefficients
        kappa = Constant(self.mesh, float(self.params.kappa))
        sigma = Constant(self.mesh, float(self.params.sigma))
        Tamb = Constant(self.mesh, float(self.params.T_ambient))

        # Base bilinear/linear forms
        a = kappa * inner(grad(u), grad(v)) * dx + sigma * inner(u, v) * dx
        L = sigma * inner(Tamb, v) * dx

        # Apply boundary conditions for each surface type
        bcs: list = []
        for surface_name, category in [
            ("wall", BndCat.WALL),
            ("roof", BndCat.ROOF),
            ("ground", BndCat.GROUND),
            ("open", BndCat.OPEN),
        ]:
            spec = self.params.get_bc_spec(surface_name)
            a, L, bcs = self._apply_bc_spec(
                V, u, v, ds_cat, self.category_markers, spec, category, a, L, bcs
            )

        # Solve
        info("UrbanHeat: Solving linear system...")
        self.solution = solve(a == L, bcs=bcs, petsc_options=self.petsc_options)
        info("UrbanHeat: Solution complete")

        # Output
        if output_path is not None:
            info(f"UrbanHeat: Saving solution to {output_path}")
            self.solution.save(output_path)

        return self.solution
