"""Interface checks without mesh assembly or numerical solver execution."""

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
import multiprocessing as mp

import numpy as np
import pytest

from dtcc_sim.helmholtz import HelmholtzParameters, HelmholtzSolver
from dtcc_sim.helmholtz import solver as api
from dtcc_sim.helmholtz._backend.params import PARAMS, Params, bind_parameters


@pytest.fixture
def mesh(tmp_path):
    path = tmp_path / "mesh.xdmf"
    path.write_text('<Xdmf><DataItem Format="HDF">mesh.h5:/Mesh</DataItem></Xdmf>')
    path.with_suffix(".h5").write_bytes(b"mesh contents")
    return path


@pytest.fixture
def fake_backend(monkeypatch):
    calls = []

    def prepare():
        calls.append(
            (
                "prepare",
                PARAMS.frequency_hz,
                PARAMS.patch_workers,
                PARAMS.n_modes_per_patch,
                PARAMS.mesh_path,
            )
        )
        path = api.Path(PARAMS.preconditioner_npz_path)
        path.write_bytes(b"offline data")
        return {}

    def solve():
        calls.append(("solve", PARAMS.frequency_hz, PARAMS.patch_workers))
        return {
            "solution": np.array([1 + 2j]),
            "iterations": 3,
            "relative_residual": 1.0e-8,
            "converged": True,
            "gmres_history": (1.0, 0.01, 1.0e-8),
        }

    @contextmanager
    def configured(params, threads):
        calls.append(("threads", threads))
        with bind_parameters(params):
            yield (
                SimpleNamespace(build_and_save_preconditioner_data=prepare),
                SimpleNamespace(solve=solve),
            )

    monkeypatch.setattr(api, "_configured_backend", configured)
    monkeypatch.setattr(api, "_runtime_versions", lambda: {"test_runtime": 1})
    return calls


def test_description_does_not_import_backend_or_create_outputs(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError("Planning must not load the numerical backend")

    monkeypatch.setattr(api, "_load_backend", forbidden)
    out = tmp_path / "case"
    sim = HelmholtzSolver(
        mesh_path=tmp_path / "not-yet-present.xdmf", output_dir=out
    )
    assert sim.describe()["parameters"]["workers"] == 1
    assert not out.exists()


def test_explicit_parameters_ignore_environment(
    monkeypatch, mesh, tmp_path, fake_backend
):
    monkeypatch.setenv("HELMHOLTZ_FREQUENCY_HZ", "999")
    monkeypatch.setenv("HELMHOLTZ_PATCH_WORKERS", "99")
    params = HelmholtzParameters(frequency_hz=5, workers=16, n_modes_per_patch=40)
    result = HelmholtzSolver(
        mesh_path=mesh, output_dir=tmp_path / "case", parameters=params
    ).run()
    assert fake_backend[1] == ("prepare", 5, 16, 40, str(mesh))
    assert fake_backend[3] == ("solve", 5, 16)
    assert result.converged and result.iterations == 3
    np.testing.assert_array_equal(result.solution, [1 + 2j])
    assert result.relative_residual == 1.0e-8


def test_parameters_are_restored_after_exception():
    with bind_parameters(Params(frequency_hz=2)):
        with pytest.raises(RuntimeError):
            with bind_parameters(Params(frequency_hz=5)):
                assert PARAMS.frequency_hz == 5
                raise RuntimeError("failed stage")
        assert PARAMS.frequency_hz == 2
    with pytest.raises(RuntimeError, match="not bound"):
        _ = PARAMS.frequency_hz


def _worker_frequency(_):
    return PARAMS.frequency_hz


def test_spawn_workers_receive_explicit_configuration():
    pytest.importorskip("dolfinx")
    pytest.importorskip("threadpoolctl")
    from dtcc_sim.helmholtz._backend.offline_setup import _initialize_patch_worker

    with bind_parameters(Params(frequency_hz=5)):
        with mp.get_context("spawn").Pool(
            1, initializer=_initialize_patch_worker, initargs=(Params(frequency_hz=7),)
        ) as pool:
            assert pool.map(_worker_frequency, [None]) == [7]
        assert PARAMS.frequency_hz == 5


@pytest.mark.parametrize("changed", ["mesh", "hdf5", "parameters", "runtime"])
def test_reject_stale_offline_data(mesh, tmp_path, fake_backend, monkeypatch, changed):
    sim = HelmholtzSolver(mesh_path=mesh, output_dir=tmp_path / "case")
    sim.prepare()
    if changed == "mesh":
        mesh.write_text(mesh.read_text().replace("/Mesh", "/Other"))
    elif changed == "hdf5":
        mesh.with_suffix(".h5").write_bytes(b"changed mesh")
    elif changed == "parameters":
        sim.parameters = replace(sim.parameters, frequency_hz=2)
    else:
        monkeypatch.setattr(api, "_runtime_versions", lambda: {"test_runtime": 2})
    with pytest.raises(ValueError, match="does not match"):
        sim.solve()
    assert not any(call[0] == "solve" for call in fake_backend)


def test_run_replaces_solver_outputs_and_preserves_other_files(mesh, tmp_path, fake_backend):
    out = tmp_path / "case"
    out.mkdir()
    (out / "solution.npy").write_bytes(b"old solution")
    (out / "keep.txt").write_text("unrelated")
    sim = HelmholtzSolver(mesh_path=mesh, output_dir=out)
    sim.run()
    sim.run()
    assert (out / "keep.txt").read_text() == "unrelated"
    assert not (out / "solution.npy").exists()  # Fake backend writes no solution file.
    assert len([call for call in fake_backend if call[0] == "solve"]) == 2


def test_same_version_partitioner_build_change_rejects_offline_data(
    mesh, tmp_path, fake_backend, monkeypatch
):
    identity = {
        "pymetis": "2025.2.2",
        "extension_sha256": "unchanged-extension",
        "metis_provider_sha256": "original-library",
    }
    monkeypatch.setattr(
        api, "_runtime_versions", lambda: {"partitioner": dict(identity)}
    )
    sim = HelmholtzSolver(mesh_path=mesh, output_dir=tmp_path / "case")
    sim.prepare()
    identity["metis_provider_sha256"] = "different-build-with-the-same-version"
    with pytest.raises(ValueError, match="does not match"):
        sim.solve()
    assert not any(call[0] == "solve" for call in fake_backend)


def test_solving_completed_case_replaces_outputs(mesh, tmp_path, fake_backend):
    sim = HelmholtzSolver(mesh_path=mesh, output_dir=tmp_path / "case")
    sim.run()
    sim.solve()
    assert len([call for call in fake_backend if call[0] == "solve"]) == 2


def test_changed_preconditioner_is_rejected_before_solve(mesh, tmp_path, fake_backend):
    sim = HelmholtzSolver(mesh_path=mesh, output_dir=tmp_path / "case")
    sim.prepare()
    sim.preconditioner_path.write_bytes(b"replaced offline data")
    with pytest.raises(ValueError, match="preconditioner checksum"):
        sim.solve()
    assert not any(call[0] == "solve" for call in fake_backend)
    assert not (sim.output_dir / "diagnostics.json").exists()


def test_legacy_preparation_requires_fresh_case(mesh, tmp_path, fake_backend):
    import json

    sim = HelmholtzSolver(mesh_path=mesh, output_dir=tmp_path / "case")
    sim.prepare()
    manifest = json.loads(sim.manifest_path.read_text())
    del manifest["preconditioner_sha256"]
    manifest["schema"] = 1
    sim.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="no integrity checksum"):
        sim.solve()
    assert not any(call[0] == "solve" for call in fake_backend)


@pytest.mark.parametrize("changed", ["mesh", "preconditioner"])
def test_inputs_changed_during_solve_cannot_complete_case(
    mesh, tmp_path, fake_backend, monkeypatch, changed
):
    sim = HelmholtzSolver(mesh_path=mesh, output_dir=tmp_path / "case")
    sim.prepare()
    configured = api._configured_backend

    @contextmanager
    def mutating_backend(params, threads):
        with configured(params, threads) as (offline, online):

            def solve():
                raw = online.solve()
                path = (
                    mesh.with_suffix(".h5")
                    if changed == "mesh"
                    else sim.preconditioner_path
                )
                path.write_bytes(b"changed during solve")
                return raw

            yield offline, SimpleNamespace(solve=solve)

    monkeypatch.setattr(api, "_configured_backend", mutating_backend)
    with pytest.raises(ValueError, match="does not match"):
        sim.solve()
    assert any(call[0] == "solve" for call in fake_backend)
    assert not (sim.output_dir / "diagnostics.json").exists()
    assert not (sim.output_dir / ".helmholtz.lock").exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"workers": 0},
        {"workers": 1.5},
        {"frequency_hz": float("nan")},
        {"overlap_layers": -1},
        {"n_modes_per_patch": 0},
        {"local_solver": "unknown"},
        {"frequency_hz": True},
        {"write_solution": "false"},
        {"one_level_only": 1},
        {"A_by_cell_tag": {1: float("nan")}},
        {"V_by_cell_tag": {1: float("inf")}},
        {"beta_by_facet_tag": {-2: "1.0"}},
        {"A_by_cell_tag": {1.5: 1.0}},
        {"A_by_cell_tag": {True: 1.0}},
        {"impedance_facet_tags": (-2, -2)},
        {"dirichlet_facet_tags": (True,)},
        {"impedance_facet_tags": (-2.5,)},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        HelmholtzParameters(**kwargs)


def test_json_coefficient_tags_and_boolean_controls_keep_their_meaning():
    import json

    params = HelmholtzParameters(
        **json.loads(
            '{"A_by_cell_tag": {"1": 2.5}, "beta_by_facet_tag": {"-2": 0.5}, '
            '"impedance_facet_tags": [-2, -3], "write_solution": false}'
        )
    )
    assert params.A_by_cell_tag == {"1": 2.5}
    assert params.beta_by_facet_tag == {"-2": 0.5}
    assert params.write_solution is False


def test_cli_is_plan_only_without_run(mesh, tmp_path, fake_backend, capsys):
    from dtcc_sim.helmholtz.__main__ import main

    assert (
        main(
            ["--mesh", str(mesh), "--output", str(tmp_path / "case"), "--workers", "16"]
        )
        == 0
    )
    assert '"workers": 16' in capsys.readouterr().out
    assert not (tmp_path / "case").exists()
    assert not fake_backend


def test_prepare_rechecks_case_after_waiting_for_backend(
    mesh, tmp_path, fake_backend, monkeypatch
):
    out = tmp_path / "case"
    configured = api._configured_backend

    @contextmanager
    def racing_backend(params, threads):
        out.mkdir(exist_ok=True)
        (out / "keep.txt").write_text("other job")
        with configured(params, threads) as backend:
            yield backend

    monkeypatch.setattr(api, "_configured_backend", racing_backend)
    HelmholtzSolver(mesh_path=mesh, output_dir=out).prepare()
    assert (out / "keep.txt").read_text() == "other job"
    assert (out / "preconditioner_data.npz").exists()
    assert not (out / ".helmholtz.lock").exists()


def test_run_keeps_case_locked_across_both_stages(mesh, tmp_path, fake_backend, monkeypatch):
    sim = HelmholtzSolver(mesh_path=mesh, output_dir=tmp_path / "case")
    configured = api._configured_backend
    checks = []

    @contextmanager
    def concurrent_backend(params, threads):
        with pytest.raises(FileExistsError, match="locked"):
            HelmholtzSolver(mesh_path=mesh, output_dir=sim.output_dir).prepare()
        checks.append(True)
        with configured(params, threads) as backend:
            yield backend

    monkeypatch.setattr(api, "_configured_backend", concurrent_backend)
    sim.run()
    assert len(checks) == 2
    assert not (sim.output_dir / ".helmholtz.lock").exists()



def test_case_lock_rejects_concurrent_writer_and_releases_after_failure(tmp_path):
    with pytest.raises(RuntimeError, match="stage failed"):
        with api._case_lease(tmp_path):
            with pytest.raises(FileExistsError, match="locked"):
                with api._case_lease(tmp_path):
                    pytest.fail("second writer acquired the case")
            raise RuntimeError("stage failed")
    assert not (tmp_path / ".helmholtz.lock").exists()


@pytest.mark.parametrize("mode", ["geometric_hats", "core_overlap_ramp"])
def test_weight_choice_recorded_and_cross_mode_cache_rejected(
    mode, mesh, tmp_path, fake_backend
):
    import json

    params = HelmholtzParameters(pou_mode=mode)
    sim = HelmholtzSolver(
        mesh_path=mesh, output_dir=tmp_path / mode, parameters=params
    )
    assert sim._params().pou_mode == mode
    assert sim.describe()["parameters"]["pou_mode"] == mode
    sim.prepare()
    assert json.loads(sim.manifest_path.read_text())["parameters"]["pou_mode"] == mode
    other = "core_overlap_ramp" if mode == "geometric_hats" else "geometric_hats"
    sim.parameters = replace(params, pou_mode=other)
    with pytest.raises(ValueError, match="does not match"):
        sim.solve()
    assert not (sim.output_dir / "diagnostics.json").exists()
    sim.parameters = params
    result = sim.solve()
    assert result.prepared_identity["parameters"]["pou_mode"] == mode


def test_weight_default_and_ambiguous_names_rejected():
    assert HelmholtzParameters().pou_mode == "core_overlap_ramp"
    for mode in ("structured_hats", "from_partition_data", "unknown", None):
        with pytest.raises(ValueError, match="pou_mode"):
            HelmholtzParameters(pou_mode=mode)


def test_weight_cli_option():
    import argparse
    from dtcc_sim.helmholtz.cli import add_solver_arguments, parameters_from_args

    parser = argparse.ArgumentParser()
    add_solver_arguments(parser)
    args = parser.parse_args(["--pou-mode", "geometric_hats"])
    assert parameters_from_args(args).pou_mode == "geometric_hats"


@pytest.mark.parametrize("legacy", [False, True])
def test_incompatible_solver_rejected_before_solve(
    mesh, tmp_path, fake_backend, monkeypatch, legacy
):
    sim = HelmholtzSolver(mesh_path=mesh, output_dir=tmp_path / "case")
    sim.prepare()
    import json

    manifest = json.loads(sim.manifest_path.read_text())
    if legacy:
        manifest.pop("solver")
        manifest["schema"] = 2
    else:
        manifest["solver"]["compatibility_revision"] += 1
    sim.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="solver implementation"):
        sim.solve()
    assert not any(call[0] == "solve" for call in fake_backend)
    assert not (sim.output_dir / "solution.npy").exists()


def test_backend_source_fingerprint_detects_change(tmp_path, monkeypatch):
    root = tmp_path / "solver"
    (root / "_backend").mkdir(parents=True)
    for name in ("parameters.py", "solver.py", "_backend/operator.py"):
        (root / name).write_text("original")
    monkeypatch.setattr(api, "__file__", str(root / "solver.py"))
    (root.parent / "coordinates.py").write_text("coordinate frame")
    before = api._solver_identity()
    (root / "_backend/operator.py").write_text("changed")
    assert api._solver_identity() != before
    before = api._solver_identity()
    (root.parent / "coordinates.py").write_text("updated frame")
    assert api._solver_identity() != before


@pytest.mark.parametrize("level, expected", [("INFO", True), ("WARNING", False)])
def test_convergence_summary_uses_logger(mesh, tmp_path, fake_backend, caplog, level, expected):
    with caplog.at_level(level, logger="dtcc-sim"):
        api._logger.addHandler(caplog.handler)
        try:
            solver = HelmholtzSolver(
                mesh_path=mesh, output_dir=tmp_path / "case",
                params=HelmholtzParameters(verbose=False),
            )
            solver.run()
        finally:
            api._logger.removeHandler(caplog.handler)
    messages = [record.getMessage() for record in caplog.records]
    assert any("Preparing two-level preconditioner" in message for message in messages) is expected
    assert any("Solving Helmholtz equation" in message for message in messages) is expected
    assert any("Helmholtz solve converged after 3 iterations" in message for message in messages) is expected
    assert any("true relative residual=1.000e-08" in message for message in messages) is expected
    assert all(record.name == "dtcc-sim" for record in caplog.records)


def test_nonconvergence_is_logged_as_error(mesh, tmp_path, fake_backend, monkeypatch, caplog):
    result_type = api.HelmholtzResult
    monkeypatch.setattr(
        api, "HelmholtzResult",
        lambda **kwargs: replace(result_type(**kwargs), converged=False),
    )
    with caplog.at_level("WARNING", logger="dtcc-sim"):
        api._logger.addHandler(caplog.handler)
        try:
            solver = HelmholtzSolver(
                mesh_path=mesh, output_dir=tmp_path / "case",
                params=HelmholtzParameters(verbose=False),
            )
            assert not solver.run().converged
        finally:
            api._logger.removeHandler(caplog.handler)
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "ERROR"
    assert "FAILED to converge after 3 iterations" in caplog.records[0].getMessage()
