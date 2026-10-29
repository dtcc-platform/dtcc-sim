"""Run timing accounting, publication and nested workflow contracts."""

import json
from types import SimpleNamespace

import pytest

from dtcc_sim import timing


def test_nested_and_repeated_phases_have_inclusive_and_self_times():
    now = [0.0]
    collector = timing.RunTimings("test", clock=lambda: now[0])
    with collector.phase("mesh"):
        now[0] = 1.0
        with collector.phase("download"):
            now[0] = 3.0
        now[0] = 4.0
        with collector.phase("download"):
            now[0] = 5.0
        now[0] = 6.0
    now[0] = 7.0
    report = collector.snapshot()
    assert report["total_s"] == 7.0
    assert report["stages"] == [
        {"path": ["mesh"], "elapsed_s": 6.0, "self_s": 3.0, "calls": 1, "failures": 0},
        {
            "path": ["mesh", "download"],
            "elapsed_s": 3.0,
            "self_s": 3.0,
            "calls": 2,
            "failures": 0,
        },
    ]


def test_nested_runs_publish_once_include_export_and_reset(tmp_path, monkeypatch):
    published = []
    monkeypatch.setattr(timing, "log_table", lambda *a, **k: published.append(a))
    now = [0.0]
    factory = timing.RunTimings
    monkeypatch.setattr(
        timing, "RunTimings", lambda name: factory(name, clock=lambda: now[0])
    )

    class Solver:
        output_dir = tmp_path
        diagnostics = {}

        @timing.timed_run
        def run(self):
            with timing.phase("solve"):
                now[0] += 2.0
            return SimpleNamespace(simulation_diagnostics={})

        @timing.timed_run
        def simulate(self):
            result = self.run()
            with timing.phase("export"):
                now[0] += 3.0
            return result

    solver = Solver()
    result = solver.simulate()
    first = json.loads((tmp_path / "timings.json").read_text())
    assert len(published) == 1
    assert first["total_s"] == 5.0
    assert result.simulation_diagnostics["run_timings"] == first
    assert solver.diagnostics["run_timings"] == first
    assert [row["path"] for row in first["stages"]] == [["solve"], ["export"]]
    solver.simulate()
    second = solver.last_timings
    assert len(published) == 2
    assert second is not first
    assert second["total_s"] == 5.0
    assert all(row["calls"] == 1 for row in second["stages"])
    assert timing._current.get() is None


def test_failed_run_preserves_error_records_partial_times_and_avoids_collectives(
    tmp_path, monkeypatch
):
    class Comm:
        rank, size = 0, 2

        def allgather(self, value):
            pytest.fail("Failure unwinding must not require other MPI ranks")

    monkeypatch.setattr(timing, "_communicator", lambda _: Comm())

    class Solver:
        output_dir = tmp_path
        diagnostics = {}

        @timing.timed_run
        def simulate(self):
            with timing.phase("download"):
                raise ValueError("upstream unavailable")

    solver = Solver()
    with pytest.raises(ValueError, match="upstream unavailable"):
        solver.simulate()
    report = json.loads((tmp_path / "timings.json").read_text())
    assert report["status"] == "failed"
    assert report["aggregation"] == "local"
    assert report["stages"][0]["failures"] == 1
    assert timing._current.get() is None


def test_rank_max_aggregation_handles_stages_present_only_on_rank_zero():
    def report(total, stages):
        return {
            "total_s": total,
            "stages": [
                dict(
                    path=[name], elapsed_s=elapsed, self_s=elapsed, calls=1, failures=0
                )
                for name, elapsed in stages
            ],
        }

    combined = timing._aggregate(
        [
            report(5.0, [("download", 2.0), ("solve", 3.0)]),
            report(6.0, [("solve", 4.0)]),
        ]
    )
    assert combined["total_s"] == 6.0
    assert combined["ranks"] == 2
    assert combined["aggregation"] == "rank_max"
    assert combined["stages"][0]["elapsed_s"] == 2.0
    assert combined["stages"][1]["elapsed_s"] == 4.0


def test_core_progress_phases_join_run_when_supported(tmp_path):
    if timing._core_phase_timing is None:
        pytest.skip("Detailed Core phases require Core phase_timing support")
    from dtcc_core.common.progress import ProgressTracker

    class Solver:
        output_dir = tmp_path
        diagnostics = {}

        @timing.timed_run
        def simulate(self):
            with timing.phase("mesh_preparation"):
                with ProgressTracker(mode="silent") as progress:
                    with progress.phase("download_pointcloud"):
                        pass
                    with progress.phase("build_mesh"):
                        pass

    solver = Solver()
    solver.simulate()
    assert [row["path"] for row in solver.last_timings["stages"]] == [
        ["mesh_preparation"],
        ["mesh_preparation", "download_pointcloud"],
        ["mesh_preparation", "build_mesh"],
    ]


def test_reporting_failure_does_not_change_success_or_original_exception(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        timing, "_finish", lambda *a: (_ for _ in ()).throw(OSError("disk full"))
    )

    class Solver:
        @timing.timed_run
        def simulate(self, fail=False):
            if fail:
                raise ValueError("invalid inputs")
            return 42

    assert Solver().simulate() == 42
    with pytest.raises(ValueError, match="invalid inputs"):
        Solver().simulate(fail=True)


def test_dataset_keyword_arguments_select_output_directory(tmp_path):
    class Dataset:
        @timing.timed_run
        def build(self, args):
            with timing.phase("data_acquisition"):
                return b"result"

    dataset = Dataset()
    assert dataset.build(args=SimpleNamespace(output_dir=tmp_path)) == b"result"
    report = json.loads((tmp_path / "timings.json").read_text())
    assert report == dataset.last_timings


def test_rank_specific_children_remain_below_their_parent_in_report():
    def row(*path):
        return dict(path=list(path), elapsed_s=1.0, self_s=1.0, calls=1, failures=0)

    combined = timing._aggregate(
        [
            dict(total_s=3.0, stages=[row("mesh"), row("solve"), row("export")]),
            dict(
                total_s=3.0,
                stages=[
                    row("mesh"),
                    row("mesh", "download"),
                    row("solve"),
                    row("export"),
                ],
            ),
        ]
    )
    assert [item["path"] for item in combined["stages"]] == [
        ["mesh"],
        ["mesh", "download"],
        ["solve"],
        ["export"],
    ]
