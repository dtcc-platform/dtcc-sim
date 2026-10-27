"""Generate or reuse a small city mesh, solve, and export DTCC results."""
from pathlib import Path
import argparse
import json
import math


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='New result directory')
    parser.add_argument('--mesh-cache', type=Path, help='Shared cache; defaults beside the result directory')
    parser.add_argument('--h', type=float, default=25.0, help='Target mesh size in metres')
    parser.add_argument('--pou-mode', choices=['geometric_hats', 'core_overlap_ramp'], default='geometric_hats')
    parser.add_argument('--run', action='store_true', help='Execute; otherwise print the request only')
    args = parser.parse_args(argv)
    if args.h <= 0 or not math.isfinite(args.h):
        parser.error('--h must be positive and finite')
    output = args.output.resolve()
    if output.exists():
        parser.error('--output must be a new directory')
    from dtcc_sim.helmholtz import HelmholtzParameters, HelmholtzResult
    parameters = HelmholtzParameters(frequency_hz=5.0, rho=1.10, n_subdomains=2,
        overlap_layers=1, oversample_layers=1, workers=2, threads_per_worker=1, pou_mode=args.pou_mode)
    request = {'h': args.h, 'bounds': [319945.962899, 6398959.716755, 320045.962899, 6399059.716755],
        'domain_height': 40.0, 'min_building_detail': 1.0,
        'mesh_options': {'flat_ground': True, 'show_footprints': False}, 'strict_live': False,
        'mesh_cache_dir': args.mesh_cache or output.parent / 'small-mesh-cache',
        'output_dir': output, 'parameters': parameters}
    print(json.dumps({k: (vars(v) if k == 'parameters' else str(v) if isinstance(v, Path) else v)
                      for k, v in request.items()}, indent=2))
    if not args.run:
        return 0
    from dtcc_core import datasets
    from dtcc_core.datasets.package import export_model_package
    import dtcc_sim.datasets  # Registers the dataset.
    mesh = datasets.helmholtz_simulation(**request)
    result = HelmholtzResult.load(output)
    if not result.converged or result.relative_residual > 1e-5:
        raise RuntimeError(f'Solve did not pass: residual {result.relative_residual}')
    for extension in ['dtcc', 'xdmf', 'vtu']:
        mesh.save(output / f'field.{extension}')
    export_model_package(mesh, output / 'field.dtccpkg', canonical=True)
    summary = {'passed': True, 'h': args.h, 'pou_mode': args.pou_mode,
        'iterations': result.iterations, 'true_relative_residual': result.relative_residual,
        'vertices': len(mesh.vertices), 'tetrahedra': len(mesh.cells),
        'exports': ['dtcc', 'xdmf', 'vtu', 'dtccpkg']}
    (output / 'example-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
