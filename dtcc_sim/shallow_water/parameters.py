"""Physical and numerical settings for a shallow-water simulation."""

from dataclasses import asdict, dataclass

from ._solver.cecilia_swe.simulator import ShallowWaterParameters as _WorkerParameters


@dataclass(frozen=True)
class ShallowWaterParameters:
    final_time: float = 0.05
    n_outputs: int = 5
    cfl: float = 0.02
    gravity: float = 9.81
    bed_from_mesh: bool = False
    water_surface: float | None = None
    compact_pulse: bool = False
    h0: float = 1.0
    amplitude: float = 0.2
    sigma: float = 0.1
    x_center: float | None = None
    y_center: float | None = None
    h_floor: float = 1e-6
    tvb_M: float = 0.5
    tvb_nu: float = 1.5

    def __post_init__(self):
        _WorkerParameters(**asdict(self))
