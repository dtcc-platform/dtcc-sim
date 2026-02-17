#!/usr/bin/env python3
"""
Smooth Field Reconstruction from Point Observations
====================================================

This demo shows how to reconstruct smooth continuous fields from sparse point
measurements using PDE-based Tikhonov regularization. The method combines data
fidelity with smoothness constraints to create physically plausible interpolations.

Mathematical Background:
-----------------------
The reconstruction minimizes the energy functional:
    E(u) = ½w Σᵢ(u(xᵢ) - yᵢ)² + ½λ∫|∇u|²dx + ½α∫(u - u_bg)²dx

where:
- u is the reconstructed field
- xᵢ, yᵢ are observation locations and values
- w controls data fidelity (how closely field matches observations)
- λ enforces smoothness (penalizes large gradients)
- α anchors field to background value (prevents unbounded growth)

Applications:
------------
- Environmental monitoring (air quality, temperature, rainfall)
- Sensor network data fusion
- Meteorological analysis
- Industrial process monitoring
- Any scenario with sparse measurements in space

Physical Setup:
--------------
In this example, we create synthetic temperature measurements at 10 random
locations and reconstruct a smooth 3D temperature field over a small urban area.
"""

import numpy as np
import dtcc_sim

# Define geographic bounds (200m x 200m area in Gothenburg)
x0 = 319995.962899
y0 = 6399009.716755
L = 200.0

bounds = (x0 - 0.5 * L, y0 - 0.5 * L, x0 + 0.5 * L, y0 + 0.5 * L)

print("=" * 70)
print("Smooth Field Reconstruction Demo")
print("=" * 70)
print(f"\nDomain: {L}m x {L}m")
print(f"Bounds: ({bounds[0]:.1f}, {bounds[1]:.1f}, {bounds[2]:.1f}, {bounds[3]:.1f})")

# Create synthetic point observations
# In a real application, these would come from sensors, measurements, etc.
np.random.seed(42)
n_observations = 10

# Generate random XY locations within the domain (slightly away from boundaries)
margin = L * 0.1  # 10% margin
point_coords = np.zeros((n_observations, 3))
point_coords[:, 0] = x0 + (np.random.rand(n_observations) - 0.5) * (L - 2 * margin)
point_coords[:, 1] = y0 + (np.random.rand(n_observations) - 0.5) * (L - 2 * margin)

# For z-coordinates: use a reasonable height above terrain
# Since we don't have terrain yet, we'll use typical urban terrain elevation ~30m
# and place sensors at 5-15m above that
terrain_elevation = 30.0  # Approximate terrain elevation for this area
point_coords[:, 2] = (
    terrain_elevation + 5.0 + np.random.rand(n_observations) * 10.0
)  # Heights: 35-45m absolute

# Generate synthetic temperature values with spatial variation
# Using a function that varies with position
point_values = 20.0 + 5.0 * np.sin((point_coords[:, 0] - x0) / L * np.pi) * np.cos(
    (point_coords[:, 1] - y0) / L * np.pi
)
# Add some random noise to make it realistic
point_values += np.random.randn(n_observations) * 0.5

print(f"\nPoint Observations: {n_observations}")
print("Location (x, y, z) -> Temperature [°C]")
print("-" * 70)
for i in range(n_observations):
    x, y, z = point_coords[i]
    val = point_values[i]
    print(f"  {i+1:2d}. ({x:7.1f}, {y:7.1f}, {z:4.1f}m) -> {val:5.2f}°C")

print(f"\nTemperature range: {point_values.min():.2f}°C to {point_values.max():.2f}°C")
print(f"Mean temperature: {point_values.mean():.2f}°C")

# Configure reconstruction parameters
params = dtcc_sim.SmoothReconstructionParameters(
    lambda_smooth=1.0,  # Smoothness weight (higher = smoother)
    alpha=1e-3,  # Background anchoring (small but nonzero)
    data_weight=100.0,  # Observation fidelity (higher = closer to data)
    background_value=20.0,  # Expected background temperature
    mesh_max_mesh_size=50.0,  # Mesh resolution (meters)
    mesh_domain_height=80.0,  # Vertical extent (meters)
    z_offset=0.0,  # No vertical offset needed
)

print("\nReconstruction Parameters:")
print(f"  λ (smoothness):        {params.lambda_smooth}")
print(f"  α (anchoring):         {params.alpha}")
print(f"  w (data weight):       {params.data_weight}")
print(f"  Background value:      {params.background_value}°C")
print(f"  Mesh size:             {params.mesh_max_mesh_size}m")

# Create simulator with synthetic data
print("\n" + "=" * 70)
print("Running Reconstruction...")
print("=" * 70)

sim = dtcc_sim.SmoothReconstructionSimulator(
    bounds=bounds,
    point_coords=point_coords,
    point_values=point_values,
    field_name="temperature",
    field_unit="°C",
    params=params,
)

# Run simulation and save output
output_path = "output/smooth_reconstruction.xdmf"
field = sim.simulate(output_path=output_path)

print("\n" + "=" * 70)
print("Reconstruction Complete!")
print("=" * 70)
print(f"\nOutput saved to: {output_path}")

# Print field information
print(f"\nReconstructed Field:")
print(f"  Name: {sim.field_name}")
print(f"  Unit: {sim.field_unit}")
print(f"  DOFs: {field.function_space.dofmap.index_map.size_global}")
print(
    f"  Range: {field.x.array.min():.2f} to {field.x.array.max():.2f} {sim.field_unit}"
)
print(f"  Mean: {field.x.array.mean():.2f} {sim.field_unit}")
print(f"  Std Dev: {np.std(field.x.array):.2f} {sim.field_unit}")

print("\nVisualization:")
print(f"  Open {output_path} in ParaView to visualize the reconstruction")
print("  The smooth field interpolates between the sparse observations while")
print("  maintaining physical plausibility and smoothness.")
