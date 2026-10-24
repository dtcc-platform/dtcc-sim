from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import math


@dataclass(frozen=True)
class Params:
    sound_speed: float = 343.0
    frequency_hz: float = 1.0

    coefficient_mode: Literal["constant", "cell_tags", "functions"] = "constant"
    A_const: float = 1.0
    V_const: float = 1.0
    beta_const: float = 1.0
    A_by_cell_tag: dict[int, float] = field(default_factory=dict)
    V_by_cell_tag: dict[int, float] = field(default_factory=dict)
    beta_by_facet_tag: dict[int, float] = field(default_factory=dict)

    mesh_source: Literal["python_builder", "external_file"] = "external_file"
    python_mesh_kind: Literal["unit_cube_uniform", "unit_square_uniform"] = (
        "unit_cube_uniform"
    )
    unit_cube_nx: int = 12
    unit_cube_ny: int = 12
    unit_cube_nz: int = 12
    unit_square_nx: int = 32
    unit_square_ny: int = 32
    unit_cube_cell_type: Literal["tetrahedron", "hexahedron"] = "tetrahedron"
    unit_square_cell_type: Literal["triangle", "quadrilateral"] = "triangle"

    mesh_path: str = ""
    mesh_name: str = "mesh"
    facet_tag_name: str = "boundary_markers"
    cell_tag_name: str = "cell_tags"

    boundary_mode: Literal["impedance_only", "mixed"] = "mixed"
    impedance_facet_tags: tuple[int, ...] = (-2, -3, -4, -5, -6)
    dirichlet_facet_tags: tuple[int, ...] = tuple()
    neumann_facet_tags: tuple[int, ...] = tuple()
    use_anchor_point: bool = False
    anchor_dof_index: int = 0

    fe_family: Literal["Lagrange"] = "Lagrange"
    fe_degree: int = 1

    partitioner: Literal["structured", "metis"] = "metis"
    nx_sub: int = 2
    ny_sub: int = 2
    nz_sub: int = 2
    n_subdomains: int | None = 16
    metis_seed: int = 0
    overlap_layers: int = 1
    oversample_layers: int = 1

    pou_mode: Literal[
        "core_overlap_ramp", "geometric_hats", "structured_hats", "from_partition_data"
    ] = "core_overlap_ramp"

    rho: float = 0.5
    eps_type: str = "krylovschur"
    eps_tol: float = 1.0e-8
    eps_max_it: int = 500
    eps_initial_dim: int = 40
    eps_growth_factor: float = 1.6
    eps_max_dim: int = 400
    aug_shift: float = 1.0e-10
    aug_mult_shift: float = 1.0

    local_solver: Literal["lu", "gmres_ilu"] = "lu"
    coarse_solver: Literal["lu", "gmres_ilu"] = "lu"
    coarse_regularization_shift: float = 0.0
    outer_solver: Literal["gmres", "fgmres"] = "gmres"
    outer_rtol: float = 1.0e-6
    outer_atol: float = 0.0
    outer_max_it: int = 300
    outer_restart: int = 100
    outer_monitor: bool = True
    allow_dense_coarse_fallback: bool = False

    rhs_mode: Literal["gaussian_point_source", "point_source", "plane_wave_exact"] = (
        "gaussian_point_source"
    )
    source_amplitude_real: float = 1.0
    source_amplitude_imag: float = 0.0
    gaussian_source_sigma: float = 5.0
    source_location_mode: Literal["manual", "mesh_bounds"] = "mesh_bounds"
    source_x: float = 0.0
    source_y: float = 0.0
    source_z: float = 0.0
    source_z_weight_min: float = 0.9
    source_z_weight_max: float = 0.1
    plane_wave_direction_x: float = 1.0
    plane_wave_direction_y: float = 0.0
    plane_wave_direction_z: float = 0.0

    data_dir: str = ""
    output_dir: str = ""
    log_dir: str = ""
    preconditioner_npz_path: str = ""
    solution_npy_path: str = ""
    solution_xdmf_path: str = ""
    gmres_history_path: str = ""

    write_solution: bool = True
    write_gmres_history: bool = True
    verbose: bool = True
    allow_replicated_offline_mode: bool = False

    patch_workers: int = 1
    threads_per_worker: int = 1
    n_modes_per_patch: int = 0
    one_level_only: bool = False
    local_evp_backend: str = "arpack_then_dense"
    arpack_ncv_min_extra: int = 20
    arpack_max_nev_frac: float = 0.60
    dense_fallback_max_dim: int = 2500

    def k(self) -> float:
        return float(2.0 * math.pi * self.frequency_hz / self.sound_speed)


from contextlib import contextmanager
from contextvars import ContextVar

_current: ContextVar[Params | None] = ContextVar("helmholtz_params", default=None)


def current_parameters() -> Params:
    params = _current.get()
    if params is None:
        raise RuntimeError("Helmholtz backend parameters are not bound")
    return params


def set_worker_parameters(params: Params):
    """Set configuration for a worker process's lifetime."""
    _current.set(params)


class _ParameterProxy:
    def __getattr__(self, name):
        return getattr(current_parameters(), name)


PARAMS = _ParameterProxy()


@contextmanager
def bind_parameters(params: Params):
    """Bind an explicit configuration to this numerical call."""
    token = _current.set(params)
    try:
        yield
    finally:
        _current.reset(token)
