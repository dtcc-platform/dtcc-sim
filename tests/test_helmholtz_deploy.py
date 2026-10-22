"""Deployment rendering checks without installing or starting services."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture
def deploy():
    path = Path(__file__).resolve().parents[1] / "deploy/helmholtz/prepare.py"
    spec = importlib.util.spec_from_file_location("prepare_helmholtz_deployment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda pid: set(range(16)), raising=False
    )
    return {
        "repo": str(Path(__file__).resolve().parents[1]),
        "python": sys.executable,
        "redis": sys.executable,
        "state_dir": str(tmp_path / "state"),
    }


def test_plan_creates_nothing_and_write_is_explicit(deploy, config, tmp_path, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "bundle"
    args = ["--config", str(path), "--output", str(output)]
    assert deploy.main(args) == 0
    plan = json.loads(capsys.readouterr().out)
    assert not plan["written"] and not plan["services_started"]
    assert not output.exists()
    assert deploy.main(args + ["--write"]) == 0
    assert len(list(output.iterdir())) == 5
    assert (output / "service.env").stat().st_mode & 0o777 == 0o600
    original = (output / "service.env").read_text()
    with pytest.raises(FileExistsError):
        deploy.main(args + ["--write"])
    assert (output / "service.env").read_text() == original


def test_units_use_localhost_and_cover_child_resources(deploy, config, tmp_path):
    files = deploy.render(config, tmp_path / "bundle")
    assert "--host 127.0.0.1 --port 18081" in files["dtcc-helmholtz-api.service"]
    assert "--bind 127.0.0.1" in files["dtcc-helmholtz-broker.service"]
    assert "--appendonly yes" in files["dtcc-helmholtz-broker.service"]
    assert "--maxmemory-policy noeviction" in files["dtcc-helmholtz-broker.service"]
    assert 'DTCC_DATASET_ALLOWLIST="helmholtz_simulation"' in files["service.env"]
    worker = files["dtcc-helmholtz-worker.service"]
    assert "CPUQuota=1600%" in worker and "MemoryMax=64G" in worker
    assert "CPUAffinity=" + " ".join(map(str, range(16))) in worker
    assert "KillMode=mixed" in worker and "TimeoutStopSec=3615" in worker
    assert "Restart=on-failure" in worker
    assert f"WorkingDirectory={config['repo']}\n" in worker
    assert (
        f"EnvironmentFile={json.dumps(str(tmp_path / 'bundle/service.env'))}\n"
        in worker
    )


def test_bundle_paths_with_spaces_are_quoted(deploy, config, tmp_path):
    repo = tmp_path / "checkout with spaces"
    (repo / "service").mkdir(parents=True)
    (repo / "service/helmholtz_worker.py").touch()
    files = deploy.render(dict(config, repo=str(repo)), tmp_path / "bundle with spaces")
    for name, contents in files.items():
        if name.endswith(".service"):
            assert f"WorkingDirectory={repo}\n" in contents
            assert (
                f'EnvironmentFile="{tmp_path / "bundle with spaces/service.env"}"\n'
                in contents
            )


@pytest.mark.parametrize(
    "changes",
    [
        {"broker_port": 18081},
        {"api_port": 99999},
        {"worker_cpus": 0},
        {"job_timeout": True},
        {"worker_memory": "infinity"},
        {"state_dir": "relative"},
        {"state_dir": "/tmp/percent%"},
        {"unexpected": "value"},
        {"worker_cpus": 17},
        {"worker_cpus": 2, "worker_cpu_ids": [0, 0]},
        {"worker_cpus": 2, "worker_cpu_ids": [0, 99]},
        {"worker_cpus": 2, "worker_cpu_ids": [0, True]},
    ],
)
def test_invalid_configurations_are_rejected(deploy, config, tmp_path, changes):
    with pytest.raises(ValueError):
        deploy.render(dict(config, **changes), tmp_path / "bundle")
