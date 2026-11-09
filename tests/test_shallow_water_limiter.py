"""Cached TVB maps agree with independent scalar least-squares limiting."""

import numpy as np
import pytest


def _scalar_reference(values, data, tvb_M, tvb_nu, size, active):
    result = values.copy()
    means = values.mean(axis=1)
    for cell, nodes in enumerate(data['cell_dofs']):
        if not active[cell] or not np.any(values[cell] != 0.):
            continue
        center = data['cell_centers'][cell]
        offsets = data['dof_coords'][nodes] - center
        midpoints = data['edge_midpoints'][cell] - center
        gradient = np.linalg.lstsq(np.column_stack([np.ones(3), offsets]), values[cell], rcond=None)[0][1:]
        adjacent = data['cell_neighbors'][cell]
        neighbor_gradient = np.zeros(2)
        if len(adjacent) >= 2:
            neighbor_gradient = np.linalg.lstsq(data['cell_centers'][adjacent]-center,
                                                means[adjacent]-means[cell], rcond=None)[0]
        a = midpoints @ gradient
        b = tvb_nu*(midpoints @ neighbor_gradient)
        limited = np.array([x if abs(x) <= tvb_M*size**2 else
                            min(x, y) if x > 0. and y > 0. else
                            max(x, y) if x < 0. and y < 0. else 0.
                            for x, y in zip(a, b)])
        if abs(sum(limited)) > 1e-14:
            positive = sum(np.maximum(limited, 0.)); negative = -sum(np.minimum(limited, 0.))
            theta_pos = min(1., negative/positive) if positive > 0. else 1.
            theta_neg = min(1., positive/negative) if negative > 0. else 1.
            limited = theta_pos*np.maximum(limited, 0.) + theta_neg*np.minimum(limited, 0.)
        gradient = np.linalg.lstsq(midpoints, limited, rcond=None)[0]
        deviation = offsets @ gradient
        result[cell] = means[cell]+deviation-deviation.mean()
    return result


@pytest.mark.usefixtures('real_worker')
@pytest.mark.parametrize('tvb_M', [0., .5, 10.])
def test_cached_limiter_matches_scalar_reference_and_preserves_means(tvb_M, monkeypatch):
    from dtcc_sim.shallow_water._solver.cecilia_swe.backend import prepare_tvb_limiter, limit_scalar_component

    centers = np.array([[0., 0.], [1., 0.], [2., 0.], [1., 1.], [2., 1.], [2., 2.], [3., 1.]])
    offsets = np.array([[-.3, -.2], [.4, -.2], [-.1, .4]])
    points = centers[:, None, :] + offsets[None, :, :]*np.arange(1, 8)[:, None, None]
    data = dict(cell_dofs=np.arange(21).reshape(7, 3), dof_coords=points.reshape(-1, 2),
                cell_centers=points.mean(axis=1),
                edge_midpoints=(points[:, [1, 2, 0]]+points[:, [2, 0, 1]])*.5,
                cell_neighbors=[np.array(n, dtype=int) for n in ([], [0], [0, 1], [1, 2, 4], [3, 5], [3, 4, 6], [4, 5])],
                gdim=2)
    values = np.random.default_rng(712).normal(size=(7, 3))
    values[0] = 0.; values[6] = .3
    active = np.ones(7, dtype=bool); active[4] = False
    expected = _scalar_reference(values, data, tvb_M, 1.5, .2, active)
    prepare_tvb_limiter(data)

    def repeated_solve(*args, **kwargs):
        raise AssertionError('Fixed geometry must not be fitted during time stepping')

    monkeypatch.setattr(np.linalg, 'lstsq', repeated_solve)
    actual = values.copy().reshape(-1)
    limit_scalar_component(actual, data, tvb_M, 1.5, .2, limit_cells=active)
    np.testing.assert_allclose(actual.reshape(7, 3), expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(actual.reshape(7, 3).mean(axis=1), values.mean(axis=1), atol=1e-15)
    np.testing.assert_array_equal(actual.reshape(7, 3)[[0, 4]], values[[0, 4]])


@pytest.mark.usefixtures('real_worker')
def test_cfl_time_step_scales_with_altitude_and_signal_speed():
    from types import SimpleNamespace
    from dtcc_sim.shallow_water._solver.cecilia_swe.backend import compute_time_step

    state = SimpleNamespace(x=SimpleNamespace(array=np.array([1., 4., 0., 2., 0., 0., 0., 0., 0.])))
    comm = SimpleNamespace(allreduce=lambda value, op: value)
    h, hu, hv = np.arange(3), np.arange(3, 6), np.arange(6, 9)
    max_speed = max(2.+np.sqrt(9.81), np.sqrt(9.81*4.))
    for cfl in (.04, .08, .24):
        actual = compute_time_step(state, h, hu, hv, 9.81, cfl, .35, comm)
        assert actual == pytest.approx(cfl*.35/max_speed)


@pytest.mark.usefixtures('real_worker')
@pytest.mark.simulation
def test_city_cfl_matches_smaller_steps_for_a_smooth_wave(tmp_path):
    from dtcc_core import RectangleMesh
    from dtcc_sim import ShallowWaterParameters, ShallowWaterSolver

    fields = []
    for cfl in (.04, .24):
        result = ShallowWaterSolver(mesh=RectangleMesh(8, 8), output_dir=tmp_path/str(cfl),
            params=ShallowWaterParameters(h0=1., amplitude=1e-4, sigma=.1,
                x_center=.5, y_center=.5, final_time=.04, n_outputs=1, cfl=cfl)).run()
        assert result.diagnostics['positive_depth']
        assert result.diagnostics['relative_mass_error'] < 1e-12
        assert result.diagnostics['rejected_attempts'] == 0
        with np.load(result.case_dir/'solver/state.npz') as raw:
            fields.append(np.column_stack([raw['state'][raw[name]]
                                           for name in ('h_dofs', 'hu_dofs', 'hv_dofs')]))
    scale = 1e-4*np.array([1., np.sqrt(9.81), np.sqrt(9.81)])
    normalized_rms = np.sqrt(np.mean(((fields[1]-fields[0])/scale)**2, axis=0))
    assert np.all(normalized_rms < .002)
