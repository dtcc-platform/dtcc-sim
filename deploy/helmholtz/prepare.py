"""Render a user-systemd deployment bundle; never install or start services."""

import argparse
import json
import os
import re
from pathlib import Path


def checked_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute() or any(char in str(path) for char in "\n\r%$"):
        raise ValueError("Paths must be absolute and contain no newline, %, or $")
    return path.resolve()


def render(config, output):
    required = {"repo", "python", "redis", "state_dir"}
    optional = {
        "broker_port",
        "api_port",
        "worker_cpus",
        "worker_cpu_ids",
        "worker_memory",
        "job_timeout",
    }
    if required - config.keys() or config.keys() - required - optional:
        raise ValueError(
            "Specify repo, python, redis, state_dir and supported options only"
        )
    paths = {key: checked_path(config[key]) for key in required}
    if not (paths["repo"] / "service/helmholtz_worker.py").is_file():
        raise ValueError("repo must contain the Helmholtz service implementation")
    for key in ("python", "redis"):
        if not paths[key].is_file() or not os.access(paths[key], os.X_OK):
            raise ValueError(f"{key} must be an executable file")
    values = {
        "broker_port": 16379,
        "api_port": 18081,
        "worker_cpus": 16,
        "worker_memory": "64G",
        "job_timeout": 3600,
    }
    values.update({key: config[key] for key in optional if key in config})
    for key in ("broker_port", "api_port", "worker_cpus", "job_timeout"):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if max(values["broker_port"], values["api_port"]) > 65535:
        raise ValueError("Ports must be at most 65535")
    if values["broker_port"] == values["api_port"]:
        raise ValueError("API and Redis need different ports")
    available = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    cpu_ids = config.get("worker_cpu_ids")
    if cpu_ids is None:
        if available is None or len(available) < values["worker_cpus"]:
            raise ValueError(
                "Specify worker_cpu_ids or render on a host with enough allowed CPUs"
            )
        cpu_ids = available[: values["worker_cpus"]]
    if not isinstance(cpu_ids, list) or len(cpu_ids) != values["worker_cpus"]:
        raise ValueError("worker_cpu_ids must list one CPU ID per worker_cpus")
    if any(
        isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in cpu_ids
    ):
        raise ValueError("CPU IDs must be nonnegative integers")
    if len(set(cpu_ids)) != len(cpu_ids) or (
        available is not None and not set(cpu_ids).issubset(available)
    ):
        raise ValueError("CPU IDs must be unique and allowed on the rendering host")
    if not isinstance(values["worker_memory"], str) or not re.fullmatch(
        r"[1-9][0-9]*[KMGT]?", values["worker_memory"]
    ):
        raise ValueError("worker_memory must be a finite systemd size, such as 64G")
    quote = lambda value: json.dumps(str(value), ensure_ascii=False)
    output = checked_path(output)
    state = paths["state_dir"]
    python = quote(paths["python"])
    broker_port = values["broker_port"]
    environment = {
        "DTCC_SERVICE_NAME": "dtcc-helmholtz",
        "DTCC_MODULE_PREFIX": "dtcc_sim",
        "DTCC_DATASET_ALLOWLIST": "helmholtz_simulation",
        "CELERY_BROKER_URL": f"redis://127.0.0.1:{broker_port}/0",
        "CELERY_RESULT_BACKEND": f"redis://127.0.0.1:{broker_port}/1",
        "SHARED_RESULTS_DIR": str(state / "shared-results"),
        "DTCC_JOB_TIMEOUT": str(values["job_timeout"]),
        "PYTHONPATH": str(paths["repo"]),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    common = f"""[Service]
Type=simple
WorkingDirectory={paths["repo"]}
EnvironmentFile={quote(output / "service.env")}
Restart=on-failure
RestartSec=5
CPUAccounting=yes
MemoryAccounting=yes
TasksAccounting=yes
"""

    def unit(description, body, broker=False):
        dependency = (
            ""
            if broker
            else "Wants=dtcc-helmholtz-broker.service\nAfter=dtcc-helmholtz-broker.service\n"
        )
        return f"""[Unit]
Description={description}
PartOf=dtcc-helmholtz.target
StartLimitIntervalSec=300
StartLimitBurst=5
{dependency}
{common}{body}
"""

    files = {
        "service.env": "".join(
            f"{key}={quote(value)}\n" for key, value in environment.items()
        ),
        "dtcc-helmholtz-broker.service": unit(
            "Private Helmholtz Redis broker",
            f"""ExecStartPre=/usr/bin/mkdir -p {quote(state / "redis")}
ExecStart={quote(paths["redis"])} --bind 127.0.0.1 --protected-mode yes --port {broker_port} --dir {quote(state / "redis")} --save "" --appendonly yes --appendfsync everysec --maxmemory 256mb --maxmemory-policy noeviction
MemoryMax=512M
TasksMax=64
TimeoutStopSec=30
""",
            broker=True,
        ),
        "dtcc-helmholtz-api.service": unit(
            "Local Helmholtz API",
            f"""ExecStart={python} -m uvicorn service.main:app --host 127.0.0.1 --port {values["api_port"]}
MemoryMax=2G
TasksMax=64
TimeoutStopSec=30
""",
        ),
        "dtcc-helmholtz-worker.service": unit(
            "Supervised Helmholtz worker",
            f"""ExecStartPre=/usr/bin/mkdir -p {quote(state / "shared-results")}
ExecStart={python} -m service.helmholtz_worker --start
CPUQuota={values["worker_cpus"] * 100}%
CPUAffinity={" ".join(str(cpu) for cpu in cpu_ids)}
MemoryMax={values["worker_memory"]}
TasksMax=256
KillMode=mixed
TimeoutStopSec={values["job_timeout"] + 15}
""",
        ),
        "dtcc-helmholtz.target": """[Unit]
Description=Local Helmholtz service stack
Wants=dtcc-helmholtz-broker.service dtcc-helmholtz-api.service dtcc-helmholtz-worker.service

[Install]
WantedBy=default.target
""",
    }
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--write", action="store_true", help="Write a new bundle directory"
    )
    args = parser.parse_args(argv)
    files = render(json.loads(args.config.read_text()), args.output)
    if args.write:
        args.output.mkdir(parents=True, exist_ok=False)
        for name, contents in files.items():
            path = args.output / name
            path.write_text(contents)
            path.chmod(0o600 if name == "service.env" else 0o644)
    print(
        json.dumps(
            {
                "written": args.write,
                "services_started": False,
                "output": str(args.output.resolve()),
                "files": files,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
