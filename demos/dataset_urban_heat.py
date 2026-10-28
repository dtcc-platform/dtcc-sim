#!/usr/bin/env python3
"""
Urban Heat Simulation: Dataset Integration
===========================================

This demo shows how to use the urban heat simulator as a dataset,
integrating with the dtcc-core dataset framework.

The UrbanHeatSimulationDataset wraps the simulation workflow in a
dataset interface, enabling:
- Automatic parameter validation via Pydantic
- JSON serialization for reproducibility
- Integration with data processing pipelines
- Consistent interface with other DTCC datasets

The dataset encapsulates:
1. Geographic bounds specification
2. Volume mesh generation
3. Heat equation solution

This is the recommended approach for production workflows and
when integrating urban heat simulation into larger analysis pipelines.
"""

import dtcc_sim

# Poseidon in Gothenburg (EPSG:3006); 200 m × 200 m domain
center_x = 319995.962899
center_y = 6399009.716755
L = 200.0

bounds = (center_x - L / 2, center_y - L / 2, center_x + L / 2, center_y + L / 2)

# Create dataset arguments with simulation parameters
dataset_args = dtcc_sim.UrbanHeatSimulationArgs(
    bounds=bounds,
    output_dir="output/dataset_urban_heat",
    wall_value=25.0,
    ground_value=20.0,
    open_value=22.0,
    T_ambient=22.0,
    mesh_max_mesh_size=12.5,
    mesh_domain_height=60.0,
)

# Build dataset (runs simulation)
dataset = dtcc_sim.UrbanHeatSimulationDataset()
print(dataset)
T = dataset.build(dataset_args)
