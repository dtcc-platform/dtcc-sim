#!/usr/bin/env python3
"""
Urban Heat Simulation: Cooling Scenario
========================================

This demo simulates night-time or winter cooling conditions in an urban environment.

Physical Setup:
- Buildings release stored heat (10°C surfaces)
- Ground is cold (0°C)
- Ambient temperature is low (5°C)
- Models heat loss from buildings to cold surroundings

The simulation solves the steady-state heat equation:
    -∇·(κ∇T) + σ(T - T_ambient) = 0

This scenario is useful for studying:
- Night-time urban cooling patterns
- Winter heating demands
- Cold air drainage in urban canyons
"""

import dtcc_sim

# Define geographic bounds (200m x 200m area in Gothenburg)
x0 = 319995.962899
y0 = 6399009.716755
L = 200.0

bounds = (x0 - 0.5 * L, y0 - 0.5 * L, x0 + 0.5 * L, y0 + 0.5 * L)

# Configure cooling scenario
params = dtcc_sim.UrbanHeatParameters(
    kappa=1.0,
    sigma=0.05,
    T_ambient=5.0,  # Cold ambient
    wall_value=10.0,  # Buildings warmer than ambient
    roof_value=10.0,
    ground_value=0.0,  # Cold ground
    open_value=5.0,
    wall_bc_type="robin",
    roof_bc_type="robin",
    ground_bc_type="dirichlet",
    open_bc_type="dirichlet",
    mesh_max_mesh_size=30.0,
    mesh_domain_height=60.0,
)

# Create simulator and run
sim = dtcc_sim.UrbanHeatSimulator(bounds=bounds, params=params)
T = sim.simulate(output_path="output/cooling.xdmf")
