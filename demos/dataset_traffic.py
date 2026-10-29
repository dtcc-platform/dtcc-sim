"""Assign road traffic using population and employment data in Gothenburg."""

import dtcc_core.datasets as datasets
import dtcc_sim.datasets

center_x = 319995.962899
center_y = 6399009.716755
L = 200.0
bounds = (center_x - L / 2, center_y - L / 2, center_x + L / 2, center_y + L / 2)

result = datasets.traffic_simulation(
    bounds=bounds,
    output_dir="output/dataset_traffic",
    trips_per_employed=8.0,
    peak_hour_factor=0.30,
    background_flow_fraction=0.05,
)
result.info()
result.plot(column="flow", linewidth=2.0)
