"""Command-line launcher; computation requires --run."""

import argparse
import json

from . import HelmholtzSimulator
from .cli import add_solver_arguments, parameters_from_args


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Configure or run a two-level Helmholtz case"
    )
    parser.add_argument("--mesh", required=True, help="Existing XDMF mesh")
    parser.add_argument("--output", required=True, help="New case directory")
    add_solver_arguments(parser)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run offline setup and solve; otherwise only print settings",
    )
    args = parser.parse_args(argv)
    params = parameters_from_args(args)
    simulator = HelmholtzSimulator(
        mesh_path=args.mesh, output_dir=args.output, parameters=params
    )
    if not args.run:
        print(json.dumps(simulator.describe(), indent=2))
        return 0
    result = simulator.run()
    print(
        json.dumps(
            {
                "converged": result.converged,
                "iterations": result.iterations,
                "relative_residual": result.relative_residual,
                "output_dir": str(result.output_dir),
            },
            indent=2,
        )
    )
    return 0 if result.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
