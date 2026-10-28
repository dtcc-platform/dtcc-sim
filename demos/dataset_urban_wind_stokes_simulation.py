#!/usr/bin/env python3
"""
Urban Wind (Stokes) via Dataset API
===================================

Runs the urban wind simulation in stationary Stokes mode through the
dataset interface and prints the concrete structure of the returned data.
"""


import numpy as np

import dtcc_sim


def summarize_result(result) -> None:
    """Print a compact summary of the returned dataset object."""
    print("\nReturned object")
    print("-" * 60)
    print(f"type: {type(result)}")

    if isinstance(result, tuple) and len(result) == 2:
        u, p = result
        print("Result is a (u, p) tuple of dolfinx Functions.")
        print(f"  u type: {type(u)}")
        print(f"  p type: {type(p)}")
        return

    if not hasattr(result, "fields"):
        print("Result has no 'fields' attribute; cannot inspect DTCC field content.")
        return

    print(f"vertices: {result.num_vertices}")
    print(f"cells:    {result.num_cells}")
    print(f"fields:   {len(result.fields)}")

    for i, field in enumerate(result.fields, start=1):
        values = np.asarray(field.values)
        print(f"\nField {i}")
        print(f"  name:  {field.name}")
        print(f"  dim:   {field.dim}")
        print(f"  unit:  {field.unit}")
        print(f"  shape: {values.shape}")
        print(f"  dtype: {values.dtype}")
        if values.size > 0:
            print(f"  min:   {values.min():.6g}")
            print(f"  max:   {values.max():.6g}")
            print(f"  mean:  {values.mean():.6g}")
            print(f"  std:   {values.std():.6g}")


def main() -> None:
    output_dir = "output/dataset_urban_wind_stokes_simulation"

    # Poseidon in Gothenburg (EPSG:3006); 200 m × 200 m domain
    center_x = 319995.962899
    center_y = 6399009.716755
    L = 200.0
    bounds = (center_x - L / 2, center_y - L / 2, center_x + L / 2, center_y + L / 2)

    args = dtcc_sim.UrbanWindSimulationArgs(
        bounds=bounds,
        output_dir=output_dir,
        equations="stokes",
        wind_speed=5.0,
        wind_dir_deg=270.0,
        inlet_profile="log_law",
        mesh_max_mesh_size=12.5,
        mesh_domain_height=80.0,
    )

    dataset = dtcc_sim.UrbanWindSimulationDataset()
    print(dataset)
    print("\nRunning urban wind dataset in Stokes mode...")
    result = dataset.build(args)

    summarize_result(result)



if __name__ == "__main__":
    main()
