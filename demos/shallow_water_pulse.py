"""Spread a water pulse over dry ground in an offline, closed unit square."""

from dtcc_core import RectangleMesh
from dtcc_sim import ShallowWaterParameters, ShallowWaterSolver

output_dir = "output/shallow_water_pulse"
mesh = RectangleMesh(12, 12)
parameters = ShallowWaterParameters(
    h0=0., amplitude=.2, sigma=.12, compact_pulse=True,
    final_time=.04, cfl=.02, n_outputs=50,
)
simulation = ShallowWaterSolver(mesh=mesh, params=parameters, output_dir=output_dir)
result = simulation.simulate()
result.info()
result.plot(
    projection="xy", field="water_depth", title=f"Water depth · t = {parameters.final_time:g} s",
    coordinate_unit="m", scale_bar=True, vmin=0., cmap="Blues", theme="light",
)
