"""
Tests for the urban wind CFD solver.

All tests are offline — no calls to remote APIs.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 1) Inlet / outlet face selection
# ---------------------------------------------------------------------------


class TestInletOutletSelection:
    """Verify that the correct bbox faces are chosen for various wind directions."""

    def _select(self, wind_dir_deg: float):
        from dtcc_sim.urban_wind import UrbanWindParameters, select_inlet_outlet

        params = UrbanWindParameters(wind_speed=5.0, wind_dir_deg=wind_dir_deg)
        return select_inlet_outlet(params)

    def test_wind_from_west(self):
        """Wind from west (270°) → flow toward east.
        Inlet = xmin (-3), outlet = xmax (-4)."""
        inlet, outlet = self._select(270.0)
        assert inlet == -3, f"Expected inlet -3, got {inlet}"
        assert outlet == -4, f"Expected outlet -4, got {outlet}"

    def test_wind_from_east(self):
        """Wind from east (90°) → flow toward west.
        Inlet = xmax (-4), outlet = xmin (-3)."""
        inlet, outlet = self._select(90.0)
        assert inlet == -4, f"Expected inlet -4, got {inlet}"
        assert outlet == -3, f"Expected outlet -3, got {outlet}"

    def test_wind_from_south(self):
        """Wind from south (180°) → flow toward north.
        Inlet = ymin (-5), outlet = ymax (-6)."""
        inlet, outlet = self._select(180.0)
        assert inlet == -5, f"Expected inlet -5, got {inlet}"
        assert outlet == -6, f"Expected outlet -6, got {outlet}"

    def test_wind_from_north(self):
        """Wind from north (0°) → flow toward south.
        Inlet = ymax (-6), outlet = ymin (-5)."""
        inlet, outlet = self._select(0.0)
        assert inlet == -6, f"Expected inlet -6, got {inlet}"
        assert outlet == -5, f"Expected outlet -5, got {outlet}"

    def test_manual_override(self):
        """Manual override bypasses auto detection."""
        from dtcc_sim.urban_wind import UrbanWindParameters, select_inlet_outlet

        params = UrbanWindParameters(
            wind_speed=5.0, wind_dir_deg=270.0,
            inlet_marker=-5, outlet_marker=-6,
        )
        inlet, outlet = select_inlet_outlet(params)
        assert inlet == -5
        assert outlet == -6


# ---------------------------------------------------------------------------
# 2) Wind vector computation
# ---------------------------------------------------------------------------


class TestWindVector:
    """Verify meteorological → math conversion of wind direction."""

    def test_from_west(self):
        from dtcc_sim.urban_wind import UrbanWindParameters

        p = UrbanWindParameters(wind_speed=1.0, wind_dir_deg=270.0)
        wx, wy = p.wind_vector_xy
        # Flow direction is east: (+1, 0)
        assert abs(wx - 1.0) < 1e-10
        assert abs(wy) < 1e-10

    def test_from_south(self):
        from dtcc_sim.urban_wind import UrbanWindParameters

        p = UrbanWindParameters(wind_speed=1.0, wind_dir_deg=180.0)
        wx, wy = p.wind_vector_xy
        # Flow direction is north: (0, +1)
        assert abs(wx) < 1e-10
        assert abs(wy - 1.0) < 1e-10

    def test_from_north(self):
        from dtcc_sim.urban_wind import UrbanWindParameters

        p = UrbanWindParameters(wind_speed=1.0, wind_dir_deg=0.0)
        wx, wy = p.wind_vector_xy
        # Flow direction is south: (0, -1)
        assert abs(wx) < 1e-10
        assert abs(wy + 1.0) < 1e-10


# ---------------------------------------------------------------------------
# 3) Inlet profile expressions
# ---------------------------------------------------------------------------


class TestInletProfiles:
    """Verify the callable inlet expressions produce correct values."""

    def test_uniform(self):
        from dtcc_sim.urban_wind import UrbanWindParameters, make_inlet_velocity_expression

        p = UrbanWindParameters(wind_speed=5.0, wind_dir_deg=270.0, inlet_profile="uniform")
        f = make_inlet_velocity_expression(p)
        x = np.array([[0.0, 1.0], [0.0, 2.0], [10.0, 20.0]])  # (3, 2)
        vals = f(x)
        assert vals.shape == (3, 2)
        np.testing.assert_allclose(vals[0, :], 5.0, atol=1e-12)  # u_x
        np.testing.assert_allclose(vals[1, :], 0.0, atol=1e-12)  # u_y
        np.testing.assert_allclose(vals[2, :], 0.0, atol=1e-12)  # u_z

    def test_power_law(self):
        from dtcc_sim.urban_wind import UrbanWindParameters, make_inlet_velocity_expression

        p = UrbanWindParameters(
            wind_speed=10.0, wind_dir_deg=270.0,
            inlet_profile="power_law", power_law_alpha=0.2, u_ref_height=10.0,
        )
        f = make_inlet_velocity_expression(p)
        z_vals = np.array([0.0, 10.0, 20.0])
        x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], z_vals])
        vals = f(x)
        # At z=10 (ref height), speed should equal U_ref
        np.testing.assert_allclose(vals[0, 1], 10.0, atol=1e-10)
        # At z=0, speed = 0
        np.testing.assert_allclose(vals[0, 0], 0.0, atol=1e-10)

    def test_log_law(self):
        from dtcc_sim.urban_wind import UrbanWindParameters, make_inlet_velocity_expression

        p = UrbanWindParameters(
            wind_speed=10.0, wind_dir_deg=270.0,
            inlet_profile="log_law", z0=0.5, u_ref_height=10.0,
        )
        f = make_inlet_velocity_expression(p)
        x = np.array([[0.0], [0.0], [10.0]])  # z = 10 m
        vals = f(x)
        # At z = u_ref_height, magnitude = U_ref
        np.testing.assert_allclose(vals[0, 0], 10.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 4) Smoke test — IPCS solver on a unit box (no buildings, synthetic markers)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def box_sim_result():
    """Run a minimal IPCS solve on a tiny box and return (u, p)."""
    import dolfinx.mesh
    from mpi4py import MPI
    from dtcc_sim.urban_wind import (
        UrbanWindSimulator,
        UrbanWindParameters,
        BndCat,
    )

    # Create a coarse box mesh  [0,2]x[0,1]x[0,1]
    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        [[0.0, 0.0, 0.0], [2.0, 1.0, 1.0]],
        [4, 2, 2],
        dolfinx.mesh.CellType.tetrahedron,
    )

    # Build synthetic facet markers matching the convention:
    #  -3=xmin, -4=xmax, -5=ymin, -6=ymax, -2=top, -1=ground (z=0)
    # No buildings (all bbox).
    fdim = mesh.topology.dim - 1
    mesh.topology.create_entities(fdim)
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)

    from dolfinx.mesh import exterior_facet_indices

    boundary_facets = exterior_facet_indices(mesh.topology)
    midpoints = dolfinx.mesh.compute_midpoints(mesh, fdim, boundary_facets)
    markers_vals = np.full(len(boundary_facets), -7, dtype=np.int32)  # default
    tol = 1e-10
    for i, mp in enumerate(midpoints):
        x, y, z = mp
        if abs(x) < tol:
            markers_vals[i] = -3  # xmin
        elif abs(x - 2.0) < tol:
            markers_vals[i] = -4  # xmax
        elif abs(y) < tol:
            markers_vals[i] = -5  # ymin
        elif abs(y - 1.0) < tol:
            markers_vals[i] = -6  # ymax
        elif abs(z) < tol:
            markers_vals[i] = -1  # ground
        elif abs(z - 1.0) < tol:
            markers_vals[i] = -2  # top

    order = np.argsort(boundary_facets)
    facet_tags = dolfinx.mesh.meshtags(
        mesh, fdim, boundary_facets[order], markers_vals[order]
    )

    params = UrbanWindParameters(
        wind_speed=1.0,
        wind_dir_deg=270.0,  # from west → flow east
        dt=0.5,
        max_steps=20,
        min_steps=5,
        steady_tolerance=1e-3,
        velocity_degree=2,
        pressure_degree=1,
        wall_model="noslip",
        inlet_profile="uniform",
    )

    sim = UrbanWindSimulator(mesh=mesh, markers=facet_tags, params=params)
    result = sim.simulate()
    return result  # tuple (u, p) since no volume_mesh_dtcc


class TestSmokeIPCS:
    """Basic sanity checks on the IPCS solver output."""

    def test_returns_tuple(self, box_sim_result):
        u, p = box_sim_result
        assert u is not None
        assert p is not None

    def test_velocity_finite(self, box_sim_result):
        u, _ = box_sim_result
        assert np.all(np.isfinite(u.x.array)), "Velocity contains non-finite values"

    def test_pressure_finite(self, box_sim_result):
        _, p = box_sim_result
        assert np.all(np.isfinite(p.x.array)), "Pressure contains non-finite values"

    def test_velocity_nonzero(self, box_sim_result):
        u, _ = box_sim_result
        assert np.max(np.abs(u.x.array)) > 0.0, "Velocity is all zero"

    def test_divergence_small(self, box_sim_result):
        """Check that div(u) is reasonably small (incompressibility)."""
        u, _ = box_sim_result
        V = u.function_space
        mesh = V.mesh
        import ufl
        from mpi4py import MPI
        from dolfinx.fem import form as _fem_form, assemble_scalar
        from dolfinx.fem import petsc as _fem_petsc

        div_u_sq = ufl.div(u) ** 2 * ufl.dx
        div_norm_sq = assemble_scalar(_fem_form(div_u_sq))
        div_norm = np.sqrt(abs(float(mesh.comm.allreduce(div_norm_sq, op=MPI.SUM))))
        # For a coarse mesh with very few steps, divergence can be large;
        # this is just a "not NaN / not astronomical" sanity check.
        assert div_norm < 200.0, f"Divergence norm too large: {div_norm}"


# ---------------------------------------------------------------------------
# 5) Poiseuille channel flow validation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def poiseuille_result():
    """Run a Poiseuille-like channel flow on a long box.

    Parabolic inlet u_x(y,z) ∝ y(1-y) * z(1-z), no-slip walls,
    p=0 at outlet.  We use uniform inlet for simplicity (Poiseuille
    profile develops naturally) and just check the solution is well-behaved.
    """
    import dolfinx.mesh
    from mpi4py import MPI
    from dtcc_sim.urban_wind import (
        UrbanWindSimulator,
        UrbanWindParameters,
    )

    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        [[0.0, 0.0, 0.0], [4.0, 1.0, 1.0]],
        [8, 3, 3],
        dolfinx.mesh.CellType.tetrahedron,
    )

    fdim = mesh.topology.dim - 1
    mesh.topology.create_entities(fdim)
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)

    from dolfinx.mesh import exterior_facet_indices

    boundary_facets = exterior_facet_indices(mesh.topology)
    midpoints = dolfinx.mesh.compute_midpoints(mesh, fdim, boundary_facets)
    markers_vals = np.full(len(boundary_facets), -7, dtype=np.int32)
    tol = 1e-10
    for i, mp in enumerate(midpoints):
        x, y, z = mp
        if abs(x) < tol:
            markers_vals[i] = -3
        elif abs(x - 4.0) < tol:
            markers_vals[i] = -4
        elif abs(y) < tol:
            markers_vals[i] = -5
        elif abs(y - 1.0) < tol:
            markers_vals[i] = -6
        elif abs(z) < tol:
            markers_vals[i] = -1
        elif abs(z - 1.0) < tol:
            markers_vals[i] = -2

    order = np.argsort(boundary_facets)
    facet_tags = dolfinx.mesh.meshtags(
        mesh, fdim, boundary_facets[order], markers_vals[order]
    )

    params = UrbanWindParameters(
        wind_speed=0.5,
        wind_dir_deg=270.0,
        dt=0.5,
        max_steps=40,
        min_steps=10,
        steady_tolerance=1e-3,
        velocity_degree=2,
        pressure_degree=1,
        wall_model="noslip",
        inlet_profile="uniform",
    )

    sim = UrbanWindSimulator(mesh=mesh, markers=facet_tags, params=params)
    result = sim.simulate()
    return result


class TestPoiseuille:
    """Check basic physical plausibility of channel flow."""

    def test_finite(self, poiseuille_result):
        u, p = poiseuille_result
        assert np.all(np.isfinite(u.x.array))
        assert np.all(np.isfinite(p.x.array))

    def test_predominant_x_velocity(self, poiseuille_result):
        """In a channel aligned with x, u_x should dominate."""
        u, _ = poiseuille_result
        V = u.function_space
        ndof = V.dofmap.index_map.size_local
        bs = V.dofmap.index_map_bs
        arr = u.x.array[: ndof * bs].reshape(ndof, bs)
        ux_max = np.max(np.abs(arr[:, 0]))
        uy_max = np.max(np.abs(arr[:, 1]))
        uz_max = np.max(np.abs(arr[:, 2]))
        # x-velocity should be dominant
        assert ux_max > 0.1 * 0.5, "u_x too small for channel flow"
        assert uy_max < ux_max, "u_y should not exceed u_x in x-aligned channel"
        assert uz_max < ux_max, "u_z should not exceed u_x in x-aligned channel"


# ---------------------------------------------------------------------------
# 6) Parameters model validation
# ---------------------------------------------------------------------------


class TestUrbanWindParameters:
    def test_default_construction(self):
        from dtcc_sim.urban_wind import UrbanWindParameters

        p = UrbanWindParameters()
        assert p.nu > 0
        assert p.nu_eff == p.nu + p.nu_t

    def test_nu_eff(self):
        from dtcc_sim.urban_wind import UrbanWindParameters

        p = UrbanWindParameters(nu=1e-5, nu_t=1e-3)
        assert abs(p.nu_eff - 1.01e-3) < 1e-10


# ---------------------------------------------------------------------------
# 7) Import smoke test
# ---------------------------------------------------------------------------


def test_import_urban_wind():
    """Ensure the package imports without errors."""
    from dtcc_sim import UrbanWindSimulator, UrbanWindParameters
    from dtcc_sim.datasets import UrbanWindSimulationArgs, UrbanWindSimulationDataset

    assert UrbanWindSimulator is not None
    assert UrbanWindParameters is not None
    assert UrbanWindSimulationArgs is not None
    assert UrbanWindSimulationDataset is not None
