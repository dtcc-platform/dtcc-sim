"""Explicit configuration for a Helmholtz simulation."""

from dataclasses import dataclass, field, fields
import math
from numbers import Real
from typing import Literal


@dataclass(frozen=True)
class HelmholtzParameters:
    frequency_hz: float = 1.0
    sound_speed: float = 343.0
    rho: float = 0.5
    n_subdomains: int = 16
    overlap_layers: int = 1
    oversample_layers: int = 1
    workers: int = 1
    threads_per_worker: int = 1
    n_modes_per_patch: int | None = None
    one_level_only: bool = False
    fe_degree: int = 1
    partitioner: Literal["metis", "structured"] = "metis"
    pou_mode: Literal["core_overlap_ramp", "geometric_hats"] = "core_overlap_ramp"
    nx_sub: int = 2
    ny_sub: int = 2
    nz_sub: int = 2
    metis_seed: int = 0
    local_solver: Literal["lu", "gmres_ilu"] = "lu"
    coarse_solver: Literal["lu", "gmres_ilu"] = "lu"
    outer_rtol: float = 1.0e-6
    outer_atol: float = 0.0
    outer_max_it: int = 300
    outer_restart: int = 100
    coarse_regularization_shift: float = 0.0
    allow_dense_coarse_fallback: bool = False
    coefficient_mode: Literal["constant", "cell_tags"] = "constant"
    A_const: float = 1.0
    V_const: float = 1.0
    beta_const: float = 1.0
    A_by_cell_tag: dict[int, float] = field(default_factory=dict)
    V_by_cell_tag: dict[int, float] = field(default_factory=dict)
    beta_by_facet_tag: dict[int, float] = field(default_factory=dict)
    boundary_mode: Literal["mixed", "impedance_only"] = "mixed"
    impedance_facet_tags: tuple[int, ...] = (-2, -3, -4, -5, -6)
    dirichlet_facet_tags: tuple[int, ...] = ()
    mesh_name: str = "mesh"
    facet_tag_name: str = "boundary_markers"
    cell_tag_name: str = "cell_tags"
    rhs_mode: Literal["gaussian_point_source", "point_source", "plane_wave_exact"] = (
        "gaussian_point_source"
    )
    gaussian_source_sigma: float = 5.0
    source_amplitude_real: float = 1.0
    source_amplitude_imag: float = 0.0
    source_location_mode: Literal["manual", "mesh_bounds"] = "mesh_bounds"
    source_x: float = 0.0
    source_y: float = 0.0
    source_z: float = 0.0
    source_z_weight_min: float = 0.9
    source_z_weight_max: float = 0.1
    plane_wave_direction_x: float = 1.0
    plane_wave_direction_y: float = 0.0
    plane_wave_direction_z: float = 0.0
    local_evp_backend: Literal["arpack", "arpack_then_dense", "dense"] = (
        "arpack_then_dense"
    )
    eps_tol: float = 1.0e-8
    eps_max_it: int = 500
    eps_initial_dim: int = 40
    eps_growth_factor: float = 1.6
    aug_shift: float = 1.0e-10
    arpack_ncv_min_extra: int = 20
    arpack_max_nev_frac: float = 0.60
    dense_fallback_max_dim: int = 2500
    write_solution: bool = True
    write_gmres_history: bool = True
    verbose: bool = True

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if item.type is float and (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{item.name} must be a finite number")
            if item.type is bool and not isinstance(value, bool):
                raise ValueError(f"{item.name} must be a boolean")
        for name in ("A_by_cell_tag", "V_by_cell_tag", "beta_by_facet_tag"):
            mapping = getattr(self, name)
            if not isinstance(mapping, dict):
                raise ValueError(f"{name} must map integer tags to finite numbers")
            for tag, value in mapping.items():
                try:
                    valid_tag = not isinstance(tag, bool) and str(int(tag)) == str(tag)
                except (TypeError, ValueError, OverflowError):
                    valid_tag = False
                if (
                    not valid_tag
                    or isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"{name} must map integer tags to finite numbers")
        for name in ("impedance_facet_tags", "dirichlet_facet_tags"):
            tags = getattr(self, name)
            if not isinstance(tags, (tuple, list)) or any(
                isinstance(tag, bool) or not isinstance(tag, int) for tag in tags
            ):
                raise ValueError(f"{name} must contain integer tags")
            if len(set(tags)) != len(tags):
                raise ValueError(f"{name} must contain unique tags")
        for name in (
            "frequency_hz",
            "sound_speed",
            "rho",
            "A_const",
            "V_const",
            "gaussian_source_sigma",
            "outer_rtol",
            "eps_tol",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "workers",
            "threads_per_worker",
            "n_subdomains",
            "fe_degree",
            "nx_sub",
            "ny_sub",
            "nz_sub",
            "outer_max_it",
            "outer_restart",
            "eps_max_it",
            "eps_initial_dim",
            "arpack_ncv_min_extra",
            "dense_fallback_max_dim",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.metis_seed, bool)
            or not isinstance(self.metis_seed, int)
            or not -1 <= self.metis_seed <= 2147483647
        ):
            raise ValueError("metis_seed must be an integer in [-1, 2147483647]")
        for name in ("overlap_layers", "oversample_layers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.n_modes_per_patch is not None:
            if (
                isinstance(self.n_modes_per_patch, bool)
                or not isinstance(self.n_modes_per_patch, int)
                or self.n_modes_per_patch <= 0
            ):
                raise ValueError("n_modes_per_patch must be a positive integer or None")
        if (
            self.outer_atol < 0
            or self.coarse_regularization_shift < 0
            or self.aug_shift < 0
        ):
            raise ValueError(
                "Absolute tolerance and regularization shifts must be nonnegative"
            )
        if self.eps_growth_factor <= 1 or not 0 < self.arpack_max_nev_frac < 1:
            raise ValueError(
                "Eigenproblem growth must exceed one and max_nev_frac must lie in (0, 1)"
            )
        choices = {
            "partitioner": ("metis", "structured"),
            "pou_mode": ("core_overlap_ramp", "geometric_hats"),
            "local_solver": ("lu", "gmres_ilu"),
            "coarse_solver": ("lu", "gmres_ilu"),
            "coefficient_mode": ("constant", "cell_tags"),
            "boundary_mode": ("mixed", "impedance_only"),
            "rhs_mode": ("gaussian_point_source", "point_source", "plane_wave_exact"),
            "source_location_mode": ("manual", "mesh_bounds"),
            "local_evp_backend": ("arpack", "arpack_then_dense", "dense"),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")

        if self.rhs_mode == "plane_wave_exact" and self.coefficient_mode != "constant":
            raise ValueError("plane_wave_exact requires constant coefficients")
        if self.rhs_mode == "plane_wave_exact" and not any(
            (
                self.plane_wave_direction_x,
                self.plane_wave_direction_y,
                self.plane_wave_direction_z,
            )
        ):
            raise ValueError("Plane-wave direction must be nonzero")
