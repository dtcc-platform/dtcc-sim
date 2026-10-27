"""Offline launcher consistency checks; numerical execution is forbidden."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

from dtcc_sim.helmholtz import HelmholtzParameters
from dtcc_sim.helmholtz.__main__ import main as mesh_main


def dataset_demo():
    path = Path(__file__).resolve().parents[1] / "demos/dataset_helmholtz.py"
    spec = importlib.util.spec_from_file_location("dataset_helmholtz_demo", path)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    return demo


def test_all_launcher_defaults_match_python_parameters(tmp_path, capsys):
    mesh_main(
        ["--mesh", str(tmp_path / "mesh.xdmf"), "--output", str(tmp_path / "case")]
    )
    direct = json.loads(capsys.readouterr().out)
    dataset_demo().main(
        [
            "--bounds",
            "0",
            "0",
            "20",
            "20",
            "--h",
            "5",
            "--output",
            str(tmp_path / "case"),
            "--mesh-cache",
            str(tmp_path / "cache"),
        ]
    )
    dataset = json.loads(capsys.readouterr().out)
    assert (
        direct["parameters"]
        == dataset["parameters"]
        == json.loads(json.dumps(asdict(HelmholtzParameters())))
    )
    assert not list(tmp_path.iterdir())


def test_dataset_launcher_exposes_preconditioner_and_resource_controls(
    tmp_path, capsys
):
    dataset_demo().main(
        [
            "--bounds",
            "0",
            "0",
            "20",
            "20",
            "--h",
            "5",
            "--output",
            str(tmp_path / "case"),
            "--mesh-cache",
            str(tmp_path / "cache"),
            "--frequency",
            "5",
            "--rho",
            "1.1",
            "--subdomains",
            "64",
            "--overlap",
            "2",
            "--oversampling",
            "3",
            "--workers",
            "16",
            "--threads-per-worker",
            "1",
            "--modes",
            "40",
        ]
    )
    parameters = json.loads(capsys.readouterr().out)["parameters"]
    assert parameters["workers"] == 16 and parameters["n_subdomains"] == 64
    assert parameters["frequency_hz"] == 5 and parameters["rho"] == 1.1
    assert parameters["overlap_layers"] == 2 and parameters["oversample_layers"] == 3
    assert (
        parameters["n_modes_per_patch"] == 40 and parameters["threads_per_worker"] == 1
    )
    assert not list(tmp_path.iterdir())


def test_export_rejects_higher_degree_before_execution(tmp_path, monkeypatch):
    import pytest
    from dtcc_sim.helmholtz.launch import main
    from dtcc_sim.helmholtz.workflow import HelmholtzWorkflow

    config = tmp_path / "case.json"
    config.write_text(
        json.dumps(
            {
                "mesh_path": "mesh.xdmf",
                "output_dir": "case",
                "parameters": {"fe_degree": 2},
            }
        )
    )
    monkeypatch.setattr(
        HelmholtzWorkflow, "run", lambda *args: pytest.fail("must not execute")
    )
    with pytest.raises(SystemExit) as error:
        main(["--config", str(config), "--run", "--export", "vtu"])
    assert error.value.code == 2
    assert not (tmp_path / "case").exists()


def test_dataset_and_workflow_resolve_identical_default_mesh_requests(tmp_path):
    from dtcc_sim.helmholtz.dataset import HelmholtzSimulationArgs
    from dtcc_sim.helmholtz.meshing import CityMeshParameters

    args = HelmholtzSimulationArgs(
        bounds=[0, 0, 20, 20],
        h=5,
        output_dir=tmp_path / "case",
        mesh_cache_dir=tmp_path / "cache",
    )
    direct = CityMeshParameters(bounds=(0, 0, 20, 20), h=5)
    assert args.city_parameters().resolved_arguments() == direct.resolved_arguments()
    assert asdict(args.parameters) == asdict(HelmholtzParameters())
    assert not list(tmp_path.iterdir())
