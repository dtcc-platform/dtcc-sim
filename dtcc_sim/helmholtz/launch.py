"""Launch a mesh-and-solver workflow from a JSON configuration."""

import argparse
import json
from pathlib import Path

from .meshing import CityMeshParameters
from .parameters import HelmholtzParameters
from .workflow import HelmholtzWorkflow


def load_workflow(config_path):
    path = Path(config_path).expanduser().resolve()
    values = json.loads(path.read_text())
    for name in ("output_dir", "mesh_cache_dir", "mesh_path"):
        if values.get(name) is not None:
            given = Path(values[name]).expanduser()
            values[name] = given if given.is_absolute() else path.parent / given
    values["parameters"] = HelmholtzParameters(**values.get("parameters", {}))
    if values.get("mesh_parameters") is not None:
        values["mesh_parameters"] = CityMeshParameters(**values["mesh_parameters"])
    return HelmholtzWorkflow(**values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--run", action="store_true", help="Generate/reuse mesh, prepare, and solve"
    )
    actions.add_argument(
        "--mesh-only", action="store_true", help="Generate/reuse mesh without solving"
    )
    parser.add_argument(
        "--export",
        choices=("vtu", "xdmf", "dtcc"),
        help="Save DTCC fields after the solve",
    )
    parser.add_argument("--field-unit", default="", help="Physical unit of u, if known")
    args = parser.parse_args(argv)
    if args.export and not args.run:
        parser.error("--export requires --run")
    workflow = load_workflow(args.config)
    if args.export and workflow.parameters.fe_degree != 1:
        parser.error("DTCC field export requires FE degree 1; no workflow was started")
    if args.mesh_only:
        print(workflow.prepare_mesh())
        return 0
    if not args.run:
        print(json.dumps(workflow.describe(), indent=2))
        return 0
    result = workflow.run()
    if args.export:
        result.save_dtcc(
            result.output_dir / ("helmholtz." + args.export), unit=args.field_unit
        )
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
