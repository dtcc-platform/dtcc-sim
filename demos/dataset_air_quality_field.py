#!/usr/bin/env python3
"""
Air Quality Field Reconstruction: Dataset Integration
======================================================

This demo shows how to use the air quality field reconstruction as a dataset,
integrating with the dtcc-core dataset framework.

The AirQualityFieldDataset combines:
1. Fetching air quality sensor data from SMHI (Swedish Meteorological and Hydrological Institute)
2. Generating a 3D urban volume mesh
3. Solving a PDE-based smooth reconstruction problem using FEniCSx

The mathematical model minimizes:
    E(u) = ½w Σᵢ(u(xᵢ) - yᵢ)² + ½λ∫|∇u|²dx + ½α∫(u - u_bg)²dx

where:
- u is the reconstructed concentration field
- xᵢ, yᵢ are sensor locations and measurements
- w controls data fidelity (how closely field matches observations)
- λ enforces smoothness (penalizes large gradients)
- α anchors field to background value (prevents unbounded growth)

This produces a smooth, continuous field that interpolates between sparse
sensor measurements while maintaining physical plausibility.

Finding Good Domains:
--------------------
Air quality sensors are sparse! To get meaningful results, you need a domain
that contains active sensors. This demo uses a 1 km x 1 km area around Gothenburg.

If no sensors are found, you'll need to experiment with different locations:
- Try urban centers (cities have more monitoring stations)
- Check https://datavardluft.smhi.se/ to see where sensors are located
- Adjust center_x, center_y in the code to move the domain
- Try a slightly larger extent (1000-2000 m) to increase chances
- Different phenomena may have different sensor coverage (NO2, PM10, O3, etc.)
"""

from pathlib import Path
import dtcc_sim

# Create output directory
Path("output").mkdir(exist_ok=True)

print("=" * 70)
print("Air Quality Field Reconstruction Demo")
print("=" * 70)

# Define a small domain to keep mesh generation fast
# Using 750m x 750m in Stockholm (has 4 NO2 sensors - verified!)
# This area was found through bisection search to have good sensor coverage
center_x = 674275.0  # Stockholm
center_y = 6581875.0
extent = 750.0  # 750m

bounds = (
    center_x - extent / 2,
    center_y - extent / 2,
    center_x + extent / 2,
    center_y + extent / 2,
)

print(f"\nDomain: {extent:.0f} m x {extent:.0f} m")
print(f"Center: ({center_x:.1f}, {center_y:.1f})")
print(f"Bounds: ({bounds[0]:.1f}, {bounds[1]:.1f}, {bounds[2]:.1f}, {bounds[3]:.1f})")
print("\nNote: This demo will fetch air quality sensor data from SMHI.")
print("      If no sensors are found in this domain, you may need to try:")
print("        - Different center coordinates (urban areas more likely)")
print("        - Slightly larger domain (1-2 km)")
print("        - Different phenomenon (NO2, PM10, O3, etc.)")

# Create dataset arguments
# We'll try NO2 (nitrogen dioxide) which is commonly monitored in urban areas
dataset_args = dtcc_sim.AirQualityFieldArgs(
    bounds=bounds,
    phenomenon="NO2",  # Nitrogen dioxide
    lambda_smooth=1.0,  # Moderate smoothing
    alpha=1e-3,  # Small anchoring weight
    data_weight=100.0,  # Moderate data fidelity
    mesh_max_mesh_size=100.0,  # Coarse mesh (100m) for stability
    mesh_domain_height=80.0,  # Standard domain height
    airquality_timeout_s=30.0,  # Longer timeout for API
    airquality_max_stations=250,  # Allow more stations
)

print(f"\nPhenomenon: {dataset_args.phenomenon}")
print(f"Reconstruction Parameters:")
print(f"  λ (smoothness):  {dataset_args.lambda_smooth}")
print(f"  α (anchoring):   {dataset_args.alpha}")
print(f"  w (data weight): {dataset_args.data_weight}")
print(f"  Mesh size:       {dataset_args.mesh_max_mesh_size} m")

# Build dataset (runs full workflow: fetch sensors, mesh, solve)
print("\n" + "=" * 70)
print("Running Air Quality Field Reconstruction...")
print("=" * 70)
print("\nStep 1: Fetching air quality sensor data from SMHI...")

dataset = dtcc_sim.AirQualityFieldDataset()

try:
    volume_mesh = dataset.build(dataset_args)

    print("\n" + "=" * 70)
    print("Reconstruction Complete!")
    print("=" * 70)

    # Print results
    if volume_mesh.fields:
        field = volume_mesh.fields[0]
        print(f"\nVolume Mesh:")
        print(f"  Vertices: {volume_mesh.num_vertices}")
        print(f"  Cells: {volume_mesh.num_cells}")

        print(f"\nReconstructed Field:")
        print(f"  Name: {field.name}")
        print(f"  Unit: {field.unit}")
        print(f"  Values: {len(field.values)} vertices")
        print(
            f"  Range: {field.values.min():.2f} - {field.values.max():.2f} {field.unit}"
        )
        print(f"  Mean: {field.values.mean():.2f} {field.unit}")
        print(f"  Std Dev: {field.values.std():.2f} {field.unit}")

        # Save to file
        output_path = "output/air_quality_field.xdmf"
        volume_mesh.save(output_path)
        print(f"\nOutput saved to: {output_path}")
        print(f"  Open in ParaView to visualize the {field.name} concentration field")
    else:
        print("\nWarning: No field data in reconstructed mesh")

except Exception as e:
    print(f"\nError: {e}")
    print("\nTroubleshooting:")
    print("  - No sensors found? Try a larger domain or different location")
    print("  - Try different phenomena: 'PM10', 'O3', 'SO2', etc.")
    print("  - Check SMHI API status: https://datavardluft.smhi.se/")
    print("  - Some sensors might be offline or have missing data")

print("\n" + "=" * 70)
print("Dataset Integration Benefits:")
print("=" * 70)
print("  ✓ Automatic parameter validation (Pydantic)")
print("  ✓ Consistent interface with other DTCC datasets")
print("  ✓ Easy integration into data processing pipelines")
print("  ✓ Combines data fetching + meshing + simulation in one call")
