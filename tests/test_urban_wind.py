"""
Tests for the urban wind CFD solver.

All tests are offline — no calls to remote APIs.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

pytestmark = [pytest.mark.simulation, pytest.mark.fenics]

pytest.importorskip("dolfinx", reason="urban wind tests require dolfinx")
from dtcc_sim import NavierStokesParameters

# ---------------------------------------------------------------------------
# 1) Inlet / outlet face selection
# ---------------------------------------------------------------------------


class TestInletOutletSelection:
    """Verify that the correct bbox faces are chosen for various wind directions."""

    def _select(self, wind_dir_deg: float):
        from dtcc_sim import NavierStokesParameters
        from dtcc_sim._flow.solver import select_inlet_outlet

        params = NavierStokesParameters(wind_speed=5.0, wind_dir_deg=wind_dir_deg)
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
        from dtcc_sim import NavierStokesParameters
        from dtcc_sim._flow.solver import select_inlet_outlet

        params = NavierStokesParameters(
            wind_speed=5.0,
            wind_dir_deg=270.0,
            inlet_marker=-5,
            outlet_marker=-6,
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
        from dtcc_sim import NavierStokesParameters

        p = NavierStokesParameters(wind_speed=1.0, wind_dir_deg=270.0)
        wx, wy = p.wind_vector_xy
        # Flow direction is east: (+1, 0)
        assert abs(wx - 1.0) < 1e-10
        assert abs(wy) < 1e-10

    def test_from_south(self):
        from dtcc_sim import NavierStokesParameters

        p = NavierStokesParameters(wind_speed=1.0, wind_dir_deg=180.0)
        wx, wy = p.wind_vector_xy
        # Flow direction is north: (0, +1)
        assert abs(wx) < 1e-10
        assert abs(wy - 1.0) < 1e-10

    def test_from_north(self):
        from dtcc_sim import NavierStokesParameters

        p = NavierStokesParameters(wind_speed=1.0, wind_dir_deg=0.0)
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
        from dtcc_sim._flow.solver import (
            make_inlet_velocity_expression,
        )

        p = NavierStokesParameters(
            wind_speed=5.0, wind_dir_deg=270.0, inlet_profile="uniform"
        )
        f = make_inlet_velocity_expression(p)
        x = np.array([[0.0, 1.0], [0.0, 2.0], [10.0, 20.0]])  # (3, 2)
        vals = f(x)
        assert vals.shape == (3, 2)
        np.testing.assert_allclose(vals[0, :], 5.0, atol=1e-12)  # u_x
        np.testing.assert_allclose(vals[1, :], 0.0, atol=1e-12)  # u_y
        np.testing.assert_allclose(vals[2, :], 0.0, atol=1e-12)  # u_z

    def test_power_law(self):
        from dtcc_sim._flow.solver import (
            make_inlet_velocity_expression,
        )

        p = NavierStokesParameters(
            wind_speed=10.0,
            wind_dir_deg=270.0,
            inlet_profile="power_law",
            power_law_alpha=0.2,
            u_ref_height=10.0,
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
        from dtcc_sim._flow.solver import (
            make_inlet_velocity_expression,
        )

        p = NavierStokesParameters(
            wind_speed=10.0,
            wind_dir_deg=270.0,
            inlet_profile="log_law",
            z0=0.5,
            u_ref_height=10.0,
        )
        f = make_inlet_velocity_expression(p)
        x = np.array([[0.0], [0.0], [10.0]])  # z = 10 m
        vals = f(x)
        # At z = u_ref_height, magnitude = U_ref
        np.testing.assert_allclose(vals[0, 0], 10.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 4) VolumeMesh output field contract
# ---------------------------------------------------------------------------


class _FakeDolfinxGeometry:
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


class _FakeDolfinxMesh:
    geometry = _FakeDolfinxGeometry()


class _FakeVolumeMesh:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


def test_volume_mesh_output_fields_have_expected_units_dims_and_lengths(monkeypatch):
    from dtcc_sim._flow.solver import _dolfinx_to_volume_mesh

    def fake_evaluate(_mesh, solution):
        if solution == "velocity":
            return np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])
        return np.array([[10.0], [20.0]])

    monkeypatch.setattr(
        "dtcc_sim._flow.solver._evaluate_function_at_vertices",
        fake_evaluate,
    )
    monkeypatch.setattr(
        "dtcc_sim._flow.solver._reorder_by_coords",
        lambda _source_keys, values, _target_keys: values,
    )

    result = _dolfinx_to_volume_mesh(
        _FakeDolfinxMesh(),
        "velocity",
        "pressure",
        _FakeVolumeMesh(),
    )

    fields = {field.name: field for field in result.fields}
    assert set(fields) == {"velocity", "pressure", "speed"}
    assert fields["velocity"].dim == 3
    assert fields["velocity"].unit == "m/s"
    assert fields["velocity"].values.shape == (2, 3)
    assert fields["pressure"].dim == 1
    assert fields["pressure"].unit == "m^2/s^2"
    assert fields["pressure"].values.size == 2
    assert fields["speed"].dim == 1
    assert fields["speed"].unit == "m/s"
    np.testing.assert_allclose(fields["speed"].values, np.array([5.0, 2.0]))


def test_volume_mesh_output_validation_rejects_inconsistent_speed():
    from dtcc_core.model import Field
    from dtcc_sim._flow.solver import _validate_wind_volume_fields

    volume_mesh = _FakeVolumeMesh()
    volume_mesh.fields = [
        Field(
            name="velocity",
            values=np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]]),
            dim=3,
            unit="m/s",
        ),
        Field(
            name="pressure",
            values=np.array([10.0, 20.0]),
            dim=1,
            unit="m^2/s^2",
        ),
        Field(
            name="speed",
            values=np.array([4.0, 2.0]),
            dim=1,
            unit="m/s",
        ),
    ]

    with pytest.raises(RuntimeError, match="speed field is inconsistent"):
        _validate_wind_volume_fields(volume_mesh)


def test_use_weather_requires_usable_wind_data(monkeypatch):
    import dtcc_core.datasets as datasets
    from dtcc_sim import NavierStokesParameters, NavierStokesSolver

    class EmptyWeather:
        def to_arrays(self, _field_name):
            return np.empty((0, 2)), np.array([])

    monkeypatch.setattr(datasets, "weather", lambda **_kwargs: EmptyWeather())

    sim = NavierStokesSolver(
        bounds=(0.0, 0.0, 1.0, 1.0),
        params=NavierStokesParameters(use_weather=True),
    )

    with pytest.raises(RuntimeError, match="use_weather=True.*wind_speed"):
        sim._maybe_fetch_weather()


@pytest.mark.parametrize("option,value", [("dt", .1), ("max_steps", 10), ("simulation_mode", "statistical_steady"), ("petsc_pressure", {})])
def test_stationary_parameters_reject_obsolete_options(option, value):
    from dtcc_sim import NavierStokesParameters
    with pytest.raises(ValueError, match="Stationary flow configuration"):
        NavierStokesParameters(**{option: value})


def test_stationary_parameter_defaults_and_validation():
    from dtcc_sim import NavierStokesParameters
    p = NavierStokesParameters(nu=1e-5, nu_t=1e-3)
    assert p.nu_eff == 1e-5 + 1e-3
    assert p.velocity_degree == 2 and p.pressure_degree == 1
    for values in [{"nu": float("nan")}, {"velocity_degree": 1}, {"nonlinear_max_it": 0}]:
        with pytest.raises(ValueError):
            NavierStokesParameters(**values)
