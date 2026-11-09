"""Small-amplitude reflecting-basin wave: python -m tools.verification.shallow_water.wave.

The cosine-series reference solves the linearized equations independently of
DG assembly. Its finite-amplitude discrepancy is checked by halving amplitude.
All errors integrate the full DG1 polynomial, not only cell-centre samples.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss


def reference(points, time, *, modes=48, order=96, sigma=.1, gravity=9.81):
    """Return unit-amplitude perturbation and momentum divided by sqrt(gH), H=1."""
    nodes, weights = leggauss(order)
    x, w = (nodes + 1) / 2, weights / 2
    k = np.arange(modes) * np.pi
    coefficients = np.cos(k[:, None] * x) @ (w * np.exp(-.5*((x-.5)/sigma)**2))
    coefficients[1:] *= 2
    amplitudes = np.outer(coefficients, coefficients)
    radius = np.hypot(k[:, None], k[None, :])
    phase = np.sqrt(gravity) * radius * time
    sine = np.divide(np.sin(phase), radius, out=np.zeros_like(radius), where=radius > 0)
    cx, cy = np.cos(points[:, :1] * k), np.cos(points[:, 1:2] * k)
    sx, sy = np.sin(points[:, :1] * k), np.sin(points[:, 1:2] * k)
    return np.column_stack((
        np.einsum('pi,ij,pj->p', cx, amplitudes*np.cos(phase), cy, optimize=True),
        np.einsum('pi,ij,pj->p', sx, amplitudes*k[:, None]*sine, cy, optimize=True),
        np.einsum('pi,ij,pj->p', cx, amplitudes*k[None, :]*sine, sy, optimize=True)))


def quadrature(coordinates, order=6):
    """Duffy-product quadrature on triangles with normalized area weights."""
    x, w = leggauss(order)
    r, s = np.meshgrid((x+1)/2, (x+1)/2, indexing='ij')
    wr, ws = np.meshgrid(w/2, w/2, indexing='ij')
    basis = np.column_stack(((1-r).ravel()*(1-s).ravel(), r.ravel(), (1-r).ravel()*s.ravel()))
    weights = (2*wr*ws*(1-r)).ravel()
    points = np.einsum('qv,cvd->cqd', basis, coordinates)
    edges = coordinates[:, 1:] - coordinates[:, :1]
    area = abs(edges[:, 0, 0]*edges[:, 1, 1]-edges[:, 0, 1]*edges[:, 1, 0])/2
    return points, basis, area[:, None]*weights


def run_case(directory, n, cfl, amplitude=1e-4):
    from dtcc_core import RectangleMesh
    from dtcc_sim import ShallowWaterParameters, ShallowWaterSolver
    from dtcc_sim.shallow_water import ShallowWaterResult
    directory = Path(directory)
    params = ShallowWaterParameters(h0=1., amplitude=amplitude, sigma=.1,
        x_center=.5, y_center=.5, final_time=.04, n_outputs=1, cfl=cfl)
    if (directory/'shallow_water.json').exists():
        result = ShallowWaterResult.load(directory)
        manifest = json.loads((result.case_dir/'run.json').read_text())
        from dtcc_sim.shallow_water.solver import _reference
        assert manifest['reference'] == _reference(), 'Use a new output directory for a changed backend'
        assert manifest['parameters'] == asdict(params), 'Saved parameters differ; use a new output directory'
    else:
        mesh = RectangleMesh(n, n)
        mesh.markers = np.full(len(mesh.faces), -1, dtype=np.int32)
        result = ShallowWaterSolver(mesh=mesh, params=params, output_dir=directory).run()
    with np.load(result.case_dir/'solver/state.npz') as raw:
        cells = raw['cell_dofs']
        assert len(cells) == 2*n*n, 'Saved mesh does not match requested refinement'
        coordinates = raw['coordinates'][cells, :2]
        state = np.stack([raw['state'][raw[key]][cells] for key in ('h_dofs','hu_dofs','hv_dofs')], axis=-1)
    points, basis, weights = quadrature(coordinates)
    normalized = state.copy()
    normalized[:, :, 0] = (state[:, :, 0]-1)/amplitude
    normalized[:, :, 1:] /= amplitude*np.sqrt(9.81)
    numerical = np.einsum('qv,cvd->cqd', basis, normalized)
    exact = reference(points.reshape(-1,2), .04).reshape(numerical.shape)
    error = np.sqrt(np.sum(weights[:, :, None]*(numerical-exact)**2, axis=(0,1)))
    reference_l2 = np.sqrt(np.sum(weights[:, :, None]*exact**2, axis=(0,1)))
    frozen = np.zeros_like(exact)
    frozen[:, :, 0] = np.exp(-.5*np.sum(((points-.5)/.1)**2, axis=-1))
    frozen_error = np.sqrt(np.sum(weights[:, :, None]*(frozen-exact)**2, axis=(0,1))) / reference_l2
    refinement = reference(points.reshape(-1,2), .04, modes=64, order=128).reshape(exact.shape)
    reference_uncertainty = float(np.max(abs(refinement-exact)))
    report = dict(n=n, cfl=cfl, amplitude=amplitude, normalized_l2=error.tolist(),
        relative_l2=(error/reference_l2).tolist(), frozen_relative_l2=frozen_error.tolist(),
        reference_uncertainty=reference_uncertainty, steps=result.diagnostics.get('steps'),
        relative_mass_error=result.diagnostics['relative_mass_error'], case_dir=str(result.case_dir))
    print(json.dumps(report), flush=True)
    return report, normalized, weights, basis, coordinates


def study(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    mesh = [run_case(output/f'mesh-{n}', n, .02)[0] for n in (8,16,24)]
    time_cases = [run_case(output/f'time-{cfl}', 12, cfl) for cfl in (.08,.04,.02)]
    differences = []
    for first, second in zip(time_cases, time_cases[1:]):
        np.testing.assert_array_equal(first[4], second[4])
        delta = np.einsum('qv,cvd->cqd', first[3], first[1]-second[1])
        differences.append(np.sqrt(np.sum(first[2][:,:,None]*delta**2,axis=(0,1))).tolist())
    half = run_case(output/'half-amplitude', 16, .02, 5e-5)
    full = run_case(output/'mesh-16', 16, .02)
    np.testing.assert_array_equal(full[4], half[4])
    delta = np.einsum('qv,cvd->cqd', full[3], full[1]-half[1])
    amplitude_difference = np.sqrt(np.sum(full[2][:,:,None]*delta**2,axis=(0,1)))
    checks = dict(finest_relative_error=bool(np.all(np.array(mesh[-1]['relative_l2']) < .25)),
        mesh_errors_decrease=bool(np.all(np.diff([r['normalized_l2'] for r in mesh],axis=0)<0)),
        timestep_differences_decrease=bool(np.all(np.array(differences[1])<np.array(differences[0]))),
        small_amplitude=bool(np.all(amplitude_difference < .01*np.array(full[0]['normalized_l2']))),
        resolved_reference=max(r['reference_uncertainty'] for r in mesh)<min(1e-6, 1e-4*min(min(r['normalized_l2']) for r in mesh)))
    report = dict(mesh=mesh, time=[c[0] for c in time_cases],
        timestep_differences=differences, amplitude_difference=amplitude_difference.tolist(),
        checks=checks, passed=all(checks.values()))
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:
        raise RuntimeError('Accuracy study failed; inspect summary.json')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output/verification/shallow-water-wave'))
    study(parser.parse_args().output)
