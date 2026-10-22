"""Generate/select a city mesh, solve, and export DTCC vertex fields."""

import argparse
from dtcc_sim.helmholtz.launch import load_workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run and save fields; otherwise print the plan",
    )
    args = parser.parse_args()
    workflow = load_workflow(args.config)
    if args.run and workflow.parameters.fe_degree != 1:
        parser.error("DTCC field export requires FE degree 1; no workflow was started")
    if not args.run:
        print(workflow.describe())
        return
    result = workflow.run()
    print("Saved for viewing:", result.save_dtcc())
    print(
        "Saved for DTCC reload:", result.save_dtcc(result.output_dir / "helmholtz.dtcc")
    )
    # Open helmholtz.vtu in a viewer and select helmholtz_amplitude.


if __name__ == "__main__":
    main()
