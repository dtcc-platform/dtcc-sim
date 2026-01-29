#!/usr/bin/env python3
"""
Urban Heat Simulation: Heatwave Scenario
=========================================

This demo simulates a summer heat wave in an urban environment.

Physical Setup:
- Buildings have elevated surface temperatures (30°C from solar heating)
- Ground is heated by solar radiation (25°C)
- Ambient air temperature is 20°C
- Higher thermal diffusivity models enhanced urban airflow

The simulation solves the steady-state heat equation:
    -∇·(κ∇T) + σ(T - T_ambient) = 0

This scenario is useful for studying:
- Urban heat island effect
- Summer thermal comfort
- Peak cooling loads
- Heat stress in urban areas
"""

import dtcc_sim

# Define geographic bounds (200m x 200m area in Gothenburg)
x0 = 319995.962899
y0 = 6399009.716755
L = 200.0

bounds = (x0 - 0.5 * L, y0 - 0.5 * L, x0 + 0.5 * L, y0 + 0.5 * L)

# Configure heat wave scenario
params = dtcc_sim.UrbanHeatParameters(
    kappa=2.0,  # Higher thermal diffusivity (enhanced mixing)
    sigma=0.1,  # Relaxation to ambient air
    T_ambient=20.0,  # Ambient temperature
    wall_value=30.0,  # Hot building surfaces
    wall_h=5.0,  # Convective heat transfer coefficient
    roof_value=30.0,  # Hot roofs (solar heated)
    roof_h=5.0,
    ground_value=25.0,  # Warm ground
    ground_h=2.0,
    open_value=20.0,  # Far-field ambient temperature
    wall_bc_type="robin",
    roof_bc_type="robin",
    ground_bc_type="robin",
    open_bc_type="dirichlet",
    mesh_max_mesh_size=30.0,
    mesh_domain_height=60.0,
)

# Create simulator and run
sim = dtcc_sim.UrbanHeatSimulator(bounds=bounds, params=params)
T = sim.simulate(output_path="demos/output/heatwave.xdmf")
