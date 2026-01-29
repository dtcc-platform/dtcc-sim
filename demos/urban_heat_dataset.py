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

from pathlib import Path
import dtcc_sim

# Create output directory
Path("output").mkdir(exist_ok=True)

# Define geographic bounds (200m x 200m area in Gothenburg)
x0 = 319995.962899
y0 = 6399009.716755
L = 200.0

bounds = (x0 - 0.5 * L, y0 - 0.5 * L, x0 + 0.5 * L, y0 + 0.5 * L)

# Create dataset arguments with simulation parameters
dataset_args = dtcc_sim.UrbanHeatSimulationArgs(
    bounds=bounds,
    wall_value=25.0,
    ground_value=20.0,
    open_value=22.0,
    T_ambient=22.0,
    mesh_max_mesh_size=30.0,
    mesh_domain_height=60.0,
)

# Build dataset (runs simulation)
dataset = dtcc_sim.UrbanHeatSimulationDataset()
print(dataset)
T = dataset.build(dataset_args)
