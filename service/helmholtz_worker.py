"""Plan a dedicated complex-PETSc worker; starting requires --start."""

import argparse
import json
import os
import shlex
import sys

from service.config import HELMHOLTZ_QUEUE, MODULE_PREFIX


def worker_command():
    return [
        sys.executable,
        "-m",
        "celery",
        "-A",
        "service.tasks:celery_app",
        "worker",
        "--queues",
        HELMHOLTZ_QUEUE,
        "--pool=solo",
        "--concurrency=1",
        "--prefetch-multiplier=1",
        "--hostname=helmholtz@%h",
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start", action="store_true", help="Check runtime and start worker"
    )
    args = parser.parse_args(argv)
    if MODULE_PREFIX != "dtcc_sim":
        parser.error("This worker requires DTCC_MODULE_PREFIX=dtcc_sim")
    command = worker_command()
    if not args.start:
        print(
            json.dumps(
                {
                    "queue": HELMHOLTZ_QUEUE,
                    "command": shlex.join(command),
                    "starts_worker": False,
                },
                indent=2,
            )
        )
        return 0

    from dtcc_sim.helmholtz.check_runtime import inspect_runtime

    # Check the ability to create patch subprocesses, not a job's worker count.
    report = inspect_runtime(workers=2, mesh_generation=True)
    if not report["ready"]:
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 1
    os.execv(sys.executable, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
