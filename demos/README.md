# DTCC Sim Demos

This directory contains demonstration scripts for the DTCC Sim urban simulation tools.

## Urban Heat Simulation Demos

Three focused demos illustrate different aspects of the urban heat simulator:

### 1. Cooling Scenario (`urban_heat_cooling.py`)

Simulates night-time or winter cooling conditions.

**Physical Setup:**
- Buildings release stored heat (10°C)
- Cold ground (0°C)
- Low ambient temperature (5°C)

**Usage:**
```bash
python urban_heat_cooling.py
```

**Output:** `demos/output/cooling.xdmf`

### 2. Heatwave Scenario (`urban_heat_heatwave.py`)

Simulates summer heat wave conditions.

**Physical Setup:**
- Hot building surfaces (30°C from solar heating)
- Warm ground (25°C)
- Elevated ambient temperature (20°C)

**Usage:**
```bash
python urban_heat_heatwave.py
```

**Output:** `demos/output/heatwave.xdmf`

### 3. Dataset Integration (`urban_heat_dataset.py`)

Demonstrates using the simulator via the dataset interface.

**Usage:**
```bash
python urban_heat_dataset.py
```

**Output:** `demos/output/dataset.xdmf`

## Physical Model

The simulator solves the steady-state heat equation with relaxation:

```
-∇·(κ∇T) + σ(T - T_ambient) = 0
```

where:
- `T`: temperature field [°C]
- `κ`: thermal diffusivity [m²/s]
- `σ`: relaxation coefficient [1/s]
- `T_ambient`: background temperature [°C]

### Boundary Conditions

Applied to different urban surfaces:
- **Walls**: Building vertical surfaces
- **Roofs**: Building tops
- **Ground**: Terrain surface
- **Open**: Far-field boundaries

## Visualization

View results using ParaView:
```bash
paraview demos/output/cooling.xdmf
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kappa` | 1.0 | Thermal diffusivity [m²/s] |
| `sigma` | 0.0 | Relaxation coefficient [1/s] |
| `T_ambient` | 0.0 | Ambient temperature [°C] |
| `wall_value` | 1.0 | Wall temperature [°C] |
| `roof_value` | 1.0 | Roof temperature [°C] |
| `ground_value` | 0.0 | Ground temperature [°C] |

See `UrbanHeatParameters` for complete parameter list.
