"""Run a JSON-configured shallow-water case."""

import argparse
import json
from pathlib import Path

from . import ShallowWaterParameters, ShallowWaterSimulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    for name in ("mesh_path", "mesh_data_path"):
        if settings.get(name):
            settings[name] = str((args.config.resolve().parent / settings[name]).resolve())
    ShallowWaterSimulator(parameters=ShallowWaterParameters(**settings), output_dir=args.output_dir).run()


if __name__ == "__main__":
    main()
