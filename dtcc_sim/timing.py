"""Wall-clock timings shared by solver and dataset entry points."""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
import json
from pathlib import Path
from time import perf_counter

from dtcc_core.common import get_python_logger, log_table

_logger = get_python_logger("dtcc-sim")
_current = ContextVar("dtcc_sim_timings", default=None)

try:
    from dtcc_core.common.progress import phase_timing as _core_phase_timing
except ImportError:
    _core_phase_timing = None


class RunTimings:
    """Accumulate nested phases using a monotonic wall clock.

    A phase's elapsed time includes children; its self time excludes them.
    Repeated paths accumulate elapsed time and call counts within one run.
    """

    def __init__(self, name, *, clock=perf_counter):
        self.name = name
        self.clock = clock
        self.started = clock()
        self.stages = {}
        self.stack = []
        self.owners = []
        self.results = []

    @contextmanager
    def phase(self, name):
        path = tuple(item[0] for item in self.stack) + (name,)
        row = self.stages.setdefault(
            path, {"elapsed_s": 0.0, "self_s": 0.0, "calls": 0, "failures": 0}
        )
        frame = [name, 0.0]
        started = self.clock()
        self.stack.append(frame)
        try:
            yield
        except BaseException:
            row["failures"] += 1
            raise
        finally:
            elapsed = self.clock() - started
            self.stack.pop()
            row["elapsed_s"] += elapsed
            row["self_s"] += max(0.0, elapsed - frame[1])
            row["calls"] += 1
            if self.stack:
                self.stack[-1][1] += elapsed

    def snapshot(self, status="completed"):
        return {
            "name": self.name,
            "status": status,
            "clock": "wall",
            "aggregation": "local",
            "ranks": 1,
            "total_s": self.clock() - self.started,
            "core_phases": _core_phase_timing is not None,
            "stages": [
                {"path": list(path), **row} for path, row in self.stages.items()
            ],
        }


def phase(name):
    """Time a phase when a solver run is active; otherwise do nothing."""
    collector = _current.get()
    return collector.phase(name) if collector is not None else nullcontext()


def timed(name):
    """Include a function in the active run's phase breakdown."""

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with phase(name):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def _communicator(collector):
    for owner in collector.owners:
        mesh = getattr(owner, "mesh", None)
        comm = getattr(mesh, "comm", None)
        if comm is not None:
            return comm
    # Use the existing MPI runtime only; importing it would initialize MPI for
    # non-FEniCS runs such as traffic assignment.
    import sys

    mpi = sys.modules.get("mpi4py.MPI")
    return (
        mpi.COMM_WORLD
        if mpi is not None and mpi.Is_initialized() and not mpi.Is_finalized()
        else None
    )


def _aggregate(reports):
    merged = dict(reports[0], aggregation="rank_max", ranks=len(reports))
    merged["total_s"] = max(report["total_s"] for report in reports)
    stages = {}
    for report in reports:
        for row in report["stages"]:
            path = tuple(row["path"])
            target = stages.setdefault(path, dict(row))
            for key in ("elapsed_s", "self_s", "calls", "failures"):
                target[key] = max(target[key], row[key])
    children = {}
    for path, row in stages.items():
        children.setdefault(path[:-1], []).append(row)

    def ordered(parent):
        for row in children.get(parent, []):
            yield row
            yield from ordered(tuple(row["path"]))

    merged["stages"] = list(ordered(()))
    return merged


def _attach(target, report):
    if target is None or isinstance(target, (str, bytes, Path, tuple)):
        return
    diagnostics = getattr(target, "diagnostics", None)
    if isinstance(diagnostics, dict):
        diagnostics["run_timings"] = report
    diagnostics = getattr(target, "simulation_diagnostics", None)
    if isinstance(diagnostics, dict):
        diagnostics["run_timings"] = report
    timings = getattr(target, "run_timings", None)
    if isinstance(timings, dict):
        timings.clear()
        timings.update(report)
    attributes = getattr(target, "attributes", None)
    if isinstance(attributes, dict):
        attributes.setdefault("simulation_diagnostics", {})["run_timings"] = report


def _finish(collector, directory, failed):
    report = collector.snapshot("failed" if failed else "completed")
    comm = _communicator(collector)
    # Failed runs may leave ranks on different control paths. Do not add a
    # collective during exception unwinding.
    if comm is not None and comm.size > 1 and not failed:
        report = _aggregate(comm.allgather(report))
    for owner in collector.owners:
        owner.last_timings = report
        _attach(owner, report)
    for result in collector.results:
        _attach(result, report)
    if comm is not None and comm.rank != 0:
        return
    rows = []
    for stage in report["stages"]:
        label = "  " * (len(stage["path"]) - 1) + stage["path"][-1].replace("_", " ")
        rows.append(
            (
                label,
                f"{stage['elapsed_s']:.3f}",
                f"{stage['self_s']:.3f}",
                str(stage["calls"]),
            )
        )
    rows.append(("Total run", f"{report['total_s']:.3f}", "", ""))
    log_table(
        _logger.info,
        f"{collector.name} timings ({report['status']}; wall seconds; parent includes children)",
        ["Step", ("Elapsed", "right"), ("Self", "right"), ("Calls", "right")],
        rows,
    )
    directory = (
        Path(directory).expanduser().resolve() if directory is not None else None
    )
    if directory is not None and directory.is_dir():
        path = directory / "timings.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(path)


def timed_run(function):
    """Collect one report for a public call and all its nested solver calls."""

    @wraps(function)
    def wrapped(owner, *args, **kwargs):
        collector = _current.get()
        root = collector is None
        if root:
            collector = RunTimings(f"{type(owner).__name__}.{function.__name__}")
        if not any(item is owner for item in collector.owners):
            collector.owners.append(owner)
        if not root:
            result = function(owner, *args, **kwargs)
            collector.results.append(result)
            return result
        directory = getattr(owner, "output_dir", None)
        if directory is None:
            request = args[0] if args else kwargs.get("args")
            directory = getattr(request, "output_dir", None)
        if directory is None and kwargs.get("output_path"):
            directory = Path(kwargs["output_path"]).expanduser().resolve().parent
        token = _current.set(collector)
        failed = True
        try:
            observer = (
                _core_phase_timing(collector.phase)
                if _core_phase_timing is not None
                else nullcontext()
            )
            with observer:
                result = function(owner, *args, **kwargs)
                collector.results.append(result)
                failed = False
                return result
        finally:
            _current.reset(token)
            try:
                _finish(collector, directory, failed)
            except Exception as exc:
                _logger.warning(f"Could not publish run timings: {exc}")

    return wrapped
