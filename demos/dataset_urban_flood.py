"""Simulate a water pulse through the experimental flat-city flood dataset."""

import dtcc_core as dtcc
import dtcc_sim.datasets
from dtcc_sim import ShallowWaterParameters

# Poseidon in Gothenburg (EPSG:3006); 200 m × 200 m domain
center_x = 319995.962899
center_y = 6399009.716755
L = 200.
bounds = (center_x - L/2, center_y - L/2, center_x + L/2, center_y + L/2)

flood = dtcc.datasets.urban_flood_simulation(
    bounds=bounds, h=10., min_building_detail=.5, strict_live=True,
    output_dir="output/dataset_urban_flood",
    parameters=ShallowWaterParameters(
        h0=0., amplitude=.2, sigma=10., compact_pulse=True,
        x_center=center_x, y_center=center_y, final_time=.5, n_outputs=4,
    ),
)
flood.info()
flood.plot(field="water_depth")
