# This demo runs a static traffic assignment for an area in Gothenburg.

import dtcc_core as dtcc
import dtcc_core.datasets as datasets
import dtcc_sim.datasets  # noqa: F401

# Center coordinates (Poseidon statue in Gothenburg)
x0 = 319995.962899
y0 = 6399009.716755

# Domain size
L = 2000.0

# Define bounds
bounds = dtcc.Bounds(x0 - 0.5 * L, y0 - 0.5 * L, x0 + 0.5 * L, y0 + 0.5 * L)

# Run traffic simulation
roads = datasets.traffic_simulation(
    bounds=bounds,
    trips_per_employed=8.0,
    peak_hour_factor=0.30,
    background_flow_fraction=0.05,
)
roads.info()

# Get data as arrays
arrays = roads.to_arrays()
attributes = arrays["attributes"]
print(f"Roads: {len(arrays['edges'])}")
print(f"Assigned flow: {attributes['flow_assigned'].sum():.0f}")
print(f"Background flow: {attributes['flow_background'].sum():.0f}")
print(f"Total flow: {attributes['flow'].sum():.0f}")
print(f"Max segment flow: {attributes['flow'].max():.0f}")
print(f"Max volume/capacity: {attributes['volume_capacity_ratio'].max():.2f}")

# Plot assigned road flow with matplotlib
roads.plot(column="flow", linewidth=2.0)

# View road traffic in dtcc-viewer if installed
roads.view()
