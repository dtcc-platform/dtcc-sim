"""Inspect or explicitly execute the registered Helmholtz dataset."""

import argparse
import json
from dtcc_sim.helmholtz.dataset import HelmholtzSimulationArgs
from dtcc_sim.helmholtz.cli import add_solver_arguments, parameters_from_args
import dtcc_core.datasets as datasets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h", required=True, type=float)
    parser.add_argument("--output", required=True)
    mesh = parser.add_mutually_exclusive_group(required=True)
    mesh.add_argument("--mesh-cache", help="Generate or reuse a DTCC city mesh")
    mesh.add_argument("--mesh", help="Use an existing XDMF file")
    parser.add_argument("--bounds", type=float, nargs=4, required=True)
    add_solver_arguments(parser)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the dataset; otherwise only print validated settings",
    )
    args = parser.parse_args(argv)
    request = HelmholtzSimulationArgs(
        bounds=args.bounds,
        h=args.h,
        output_dir=args.output,
        mesh_cache_dir=args.mesh_cache,
        mesh_path=args.mesh,
        parameters=parameters_from_args(args),
    )
    print(json.dumps(request.model_dump(mode="json"), indent=2))
    if not args.run:
        return 0
    result = datasets.helmholtz_simulation(**request.model_dump(mode="python"))
    result.save(request.output_dir / "helmholtz.dtcc")
    print("Saved DTCC fields; diagnostics:", request.output_dir / "diagnostics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
