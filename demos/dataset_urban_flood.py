"""Simulate a water pulse over measured terrain between city buildings."""

import dtcc_core as dtcc
import dtcc_sim.datasets
from dtcc_sim import ShallowWaterParameters

# Chalmers Johanneberg (EPSG:3006); 250 m × 250 m domain
center_x = 319970.
center_y = 6397910.
L = 250.
bounds = (center_x - L/2, center_y - L/2, center_x + L/2, center_y + L/2)
pulse_x = center_x + 35.
pulse_y = center_y - 20.
output_dir = "output/dataset_urban_flood"
T = 100.

flood = dtcc.datasets.urban_flood_simulation(
    bounds=bounds, h=10., min_building_detail=.5, raster_cell_size=2., strict_live=True,
    output_dir=output_dir,
    parameters=ShallowWaterParameters(
        h0=0., amplitude=.2, sigma=7., compact_pulse=True, bed_from_mesh=True,
        x_center=pulse_x, y_center=pulse_y, final_time=T, cfl=.12, n_outputs=100, max_steps=None,
    ),
)
flood.info()
flood.plot(
    projection="xy", field="water_depth", context_mask=flood.markers >= 0,
    title=f"Water depth · t = {T:g} s", coordinate_unit="m", scale_bar=True,
    xlabel="Easting [m]", ylabel="Northing [m]", vmin=0., cmap="Blues", theme="light",
)
