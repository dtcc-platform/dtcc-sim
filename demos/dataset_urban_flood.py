"""Simulate a water pulse through the experimental flat-city flood dataset."""

import dtcc_core as dtcc
import dtcc_sim.datasets
from dtcc_sim import ShallowWaterParameters
from dtcc_sim.shallow_water import ShallowWaterResult

# Chalmers Johanneberg (EPSG:3006); 250 m × 250 m domain
center_x = 319970.
center_y = 6397910.
L = 250.
bounds = (center_x - L/2, center_y - L/2, center_x + L/2, center_y + L/2)
pulse_x = center_x + 35.
pulse_y = center_y - 20.
output_dir = "output/dataset_urban_flood"

flood = dtcc.datasets.urban_flood_simulation(
    bounds=bounds, h=10., min_building_detail=.5, strict_live=True,
    output_dir=output_dir,
    parameters=ShallowWaterParameters(
        h0=0., amplitude=.2, sigma=7., compact_pulse=True,
        x_center=pulse_x, y_center=pulse_y, final_time=20., cfl=.08, n_outputs=4,
    ),
)
flood.info()
ShallowWaterResult.load(output_dir).plot()
