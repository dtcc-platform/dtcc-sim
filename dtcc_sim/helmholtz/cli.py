"""Shared solver options for the Helmholtz command-line entry points."""

from .parameters import HelmholtzParameters


_OPTIONS = (
    ("--frequency", "frequency_hz", float),
    ("--rho", "rho", float),
    ("--pou-mode", "pou_mode", str),
    ("--subdomains", "n_subdomains", int),
    ("--overlap", "overlap_layers", int),
    ("--oversampling", "oversample_layers", int),
    ("--workers", "workers", int),
    ("--threads-per-worker", "threads_per_worker", int),
    ("--modes", "n_modes_per_patch", int),
)


def add_solver_arguments(parser):
    """Use the Python parameter defaults for every common solver CLI option."""
    defaults = HelmholtzParameters()
    for flag, name, kind in _OPTIONS:
        parser.add_argument(flag, dest=name, type=kind, default=getattr(defaults, name))


def parameters_from_args(args):
    return HelmholtzParameters(**{name: getattr(args, name) for _, name, _ in _OPTIONS})
