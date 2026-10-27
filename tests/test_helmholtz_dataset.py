"""Offline dataset registration and dispatch checks; no meshing or solver runs."""

from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError
from dtcc_core.model import VolumeMesh, Field
import dtcc_core.datasets as datasets
import dtcc_sim.datasets  # noqa: F401 - Register simulation datasets.
from dtcc_sim.helmholtz.dataset import (
    HelmholtzSimulationArgs,
    HelmholtzSimulationDataset,
)


@pytest.fixture
def run_args(tmp_path):
    return dict(
        bounds=[0, 0, 20, 20],
        h=5,
        output_dir=tmp_path / "case",
        mesh_cache_dir=tmp_path / "cache",
        parameters={
            "frequency_hz": 5,
            "workers": 16,
            "n_subdomains": 64,
            "overlap_layers": 1,
            "oversample_layers": 3,
            "n_modes_per_patch": 40,
        },
    )


@pytest.fixture
def fake_workflow(monkeypatch):
    calls = {}
    mesh = VolumeMesh(
        vertices=np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1]]),
        cells=np.array([[0, 1, 2, 3]], dtype=np.int64),
    )

    def to_dtcc(*, unit):
        calls["unit"] = unit
        mesh.fields = [
            Field(
                name=name,
                association="vertex",
                values=np.ones((4, 1)),
                dim=1,
                unit=unit,
            )
            for name in ("helmholtz_real", "helmholtz_imaginary", "helmholtz_amplitude")
        ]
        return mesh

    result = SimpleNamespace(
        converged=True,
        iterations=3,
        relative_residual=1.0e-7,
        gmres_history=(0.1, 0.001, 1.0e-7),
        output_dir=None,
        to_dtcc=to_dtcc,
        prepared_identity={
            "inputs": {"mesh.xdmf": "test-sha256"},
            "runtime": {"test_runtime": 1},
        },
    )

    class Workflow:
        def __init__(self, **kwargs):
            calls.update(kwargs)
            self.mesh_record = {"source": "test-mesh", "h": 5}
            result.output_dir = kwargs["output_dir"]

        def run(self):
            calls["ran"] = True
            return result

    monkeypatch.setattr("dtcc_sim.helmholtz.workflow.HelmholtzWorkflow", Workflow)
    return calls, result, mesh


def test_registration_and_description_do_not_execute(monkeypatch, run_args):
    monkeypatch.setattr(
        "dtcc_sim.helmholtz.workflow.HelmholtzWorkflow.run",
        lambda *args: pytest.fail("must not execute"),
    )
    descriptor = datasets.helmholtz_simulation
    description = descriptor.describe()
    assert description["data_category"] == "simulation"
    assert description["python_return_type"] == "dtcc_core.model.VolumeMesh"
    assert set(description["supported_formats"]) == {"dtccpkg", "dtcc", "vtu"}
    schema = description["args_schema"]
    assert {"h", "output_dir"} <= set(schema["required"])
    assert "parameters" in schema["properties"]
    context = descriptor.create_context(descriptor.validate(run_args))
    manifest = context.manifest()
    assert manifest.request.parameters["h"] == 5
    assert manifest.request.parameters["parameters"]["workers"] == 16
    assert (
        manifest.presentation.view_hints["default_color_attribute"]
        == "helmholtz_amplitude"
    )
    assert (
        not run_args["output_dir"].exists() and not run_args["mesh_cache_dir"].exists()
    )


@pytest.mark.parametrize(
    "mesh_crs,expected", [(None, []), ("EPSG:3006", ["EPSG:3006"])]
)
def test_supplied_mesh_crs_is_declared_not_assumed(run_args, mesh_crs, expected):
    request = dict(run_args)
    request.pop("mesh_cache_dir")
    request.update(mesh_path="local-box.xdmf", mesh_crs=mesh_crs)
    descriptor = datasets.helmholtz_simulation
    context = descriptor.create_context(descriptor.validate(request))
    assert context.metadata.crs == expected
    assert context.request.parameters["mesh_crs"] == mesh_crs


def test_generated_mesh_context_uses_epsg3006(run_args):
    descriptor = datasets.helmholtz_simulation
    assert descriptor.create_context(descriptor.validate(run_args)).metadata.crs == [
        "EPSG:3006"
    ]
    with pytest.raises(ValueError, match="supplied XDMF"):
        descriptor.validate(dict(run_args, mesh_crs="EPSG:4326"))


def test_generated_mesh_dispatch_and_native_fields(run_args, fake_workflow):
    calls, result, mesh = fake_workflow
    returned = datasets.helmholtz_simulation(
        **run_args, strict_live=True, field_unit="Pa"
    )
    assert isinstance(returned, VolumeMesh)
    assert calls["ran"] and calls["parameters"].workers == 16
    assert calls["parameters"].n_subdomains == 64
    assert calls["parameters"].n_modes_per_patch == 40
    assert calls["mesh_parameters"].h == 5
    assert calls["mesh_parameters"].options["strict_live"] is True
    assert calls["output_dir"] == run_args["output_dir"]
    assert len(returned.fields) == 3 and all(
        field.unit == "Pa" for field in returned.fields
    )
    assert not run_args["output_dir"].exists()  # The workflow is mocked.


def test_explicit_mesh_passes_declared_h(run_args, fake_workflow):
    run_args.pop("mesh_cache_dir")
    run_args["mesh_path"] = "provided.xdmf"
    HelmholtzSimulationDataset().build(HelmholtzSimulationArgs(**run_args))
    calls, _, _ = fake_workflow
    assert calls["mesh_path"].name == "provided.xdmf" and calls["mesh_h"] == 5
    assert "mesh_parameters" not in calls


@pytest.mark.parametrize("format", ["dtcc", "vtu"])
def test_single_file_serialization_delegates(
    run_args, fake_workflow, monkeypatch, format
):
    descriptor = HelmholtzSimulationDataset()

    def export(mesh, fmt):
        assert mesh is fake_workflow[2] and fmt == format
        assert len(mesh.fields) == 3
        return b"exported fields"

    monkeypatch.setattr(descriptor, "export_to_bytes", export)
    assert (
        descriptor.build(HelmholtzSimulationArgs(**run_args, format=format))
        == b"exported fields"
    )


@pytest.mark.parametrize(
    "update",
    [
        {"h": 0},
        {"h": float("nan")},
        {"bounds": [0, 0, 0, 1, 1, 1]},
        {"parameters": {"workers": 0}},
        {"parameters": {"fe_degree": 2}},
        {"mesh_options": {"max_mesh_size": 3}},
        {"mesh_options": {"unknown_option": 1}},
        {"mesh_options": {"strict_live": True}},
        {"format": "xdmf"},
        {"unexpected": True},
    ],
)
def test_invalid_arguments_rejected_before_dispatch(run_args, fake_workflow, update):
    run_args.update(update)
    with pytest.raises((ValidationError, ValueError)):
        datasets.helmholtz_simulation(**run_args)
    assert "ran" not in fake_workflow[0]


@pytest.mark.parametrize("missing", ["h", "output_dir", "mesh_cache_dir"])
def test_required_run_settings(run_args, missing):
    run_args.pop(missing)
    with pytest.raises(ValidationError):
        HelmholtzSimulationArgs(**run_args)


def test_failed_convergence_is_not_returned_as_a_field(run_args, fake_workflow):
    fake_workflow[1].converged = False
    with pytest.raises(RuntimeError, match="did not converge"):
        datasets.helmholtz_simulation(**run_args)
    assert "unit" not in fake_workflow[0]


def test_daemonic_worker_rejected_before_start(run_args, fake_workflow, monkeypatch):
    import multiprocessing

    monkeypatch.setattr(
        multiprocessing, "current_process", lambda: SimpleNamespace(daemon=True)
    )
    with pytest.raises(RuntimeError, match="non-daemonic"):
        datasets.helmholtz_simulation(**run_args)
    assert "ran" not in fake_workflow[0]


def test_example_plans_without_execution(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "demos/dataset_helmholtz.py"
    spec = importlib.util.spec_from_file_location("dataset_helmholtz_demo", path)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    monkeypatch.setattr(
        datasets,
        "helmholtz_simulation",
        lambda **kwargs: pytest.fail("must not execute"),
    )
    assert (
        demo.main(
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
        == 0
    )
    assert not list(tmp_path.iterdir())


def _completed_run(context):
    return next(
        step
        for step in context.provenance.processing_steps
        if isinstance(step, dict) and step.get("name") == "Helmholtz completed run"
    )


def test_completed_context_carries_diagnostics_and_identity(run_args, fake_workflow):
    mesh = datasets.helmholtz_simulation(**run_args, field_unit="Pa")
    context = mesh.dataset_context
    run = _completed_run(context)
    assert run["converged"] and run["iterations"] == 3
    assert run["true_relative_residual"] == 1e-7
    assert run["prepared_identity"]["inputs"]["mesh.xdmf"] == "test-sha256"
    assert run["mesh_provenance"]["h"] == 5
    assert context.request.parameters["h"] == 5
    assert context.request.parameters["parameters"]["workers"] == 16
    assert context.provenance.generated_by["package"] == "dtcc-sim"
    assert all(
        field["unit"] == "Pa" for field in context.presentation.view_hints["fields"]
    )
    manifest = context.manifest()
    assert _completed_run(manifest) == run


def test_dataset_package_bytes_preserve_completed_run(
    run_args, fake_workflow, tmp_path
):
    from dtcc_core.datasets import load_model_package

    encoded = datasets.helmholtz_simulation(**run_args, format="dtccpkg")
    path = tmp_path / "result.dtccpkg"
    path.write_bytes(encoded)
    mesh = load_model_package(path)
    run = _completed_run(mesh.dataset_context)
    assert run["iterations"] == 3 and run["true_relative_residual"] == 1e-7
    assert mesh.dataset_context.request.parameters["h"] == 5
    assert len(mesh.fields) == 3


def test_context_metadata_is_isolated_between_requests(run_args, fake_workflow):
    descriptor = HelmholtzSimulationDataset()
    args = descriptor.validate(run_args)
    descriptor.build(args)
    assert _completed_run(descriptor.create_context(args))["iterations"] == 3
    fresh = descriptor.create_context(descriptor.validate(run_args))
    assert not any(
        isinstance(step, dict) and step.get("name") == "Helmholtz completed run"
        for step in fresh.provenance.processing_steps
    )
    assert "_run_metadata" not in args.model_dump()
    assert "_run_metadata" not in descriptor.describe()["args_schema"]["properties"]
    assert all(field["unit"] == "" for field in descriptor.view_hints["fields"])


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0])
def test_invalid_diagnostics_not_exported(run_args, fake_workflow, invalid):
    fake_workflow[1].relative_residual = invalid
    with pytest.raises(RuntimeError, match="invalid true residual"):
        datasets.helmholtz_simulation(**run_args)
    assert "unit" not in fake_workflow[0]


def test_standalone_dtcc_remains_a_fields_only_export(run_args, fake_workflow):
    raw = datasets.helmholtz_simulation(**run_args, format="dtcc")
    mesh = VolumeMesh()
    mesh.from_proto(raw)
    assert len(mesh.fields) == 3
    assert mesh.dataset_context is None
