from __future__ import annotations

from pathlib import Path

from dtcc_sim.qa import (
    SIMULATION_DATASET_NAMES,
    audit_simulation_dataset_contracts,
    simulation_dataset_descriptors,
    simulation_validation_cases,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_simulation_descriptors_create_context_offline():
    descriptors = simulation_dataset_descriptors()

    assert set(descriptors) == set(SIMULATION_DATASET_NAMES)
    for name, descriptor in descriptors.items():
        args = descriptor.validate({"bounds": (0.0, 0.0, 1.0, 1.0)})
        context = descriptor.create_context(args)

        assert context.request.dataset_name == name
        assert context.request.bounds == [0.0, 0.0, 1.0, 1.0]
        assert context.metadata.data_category == "simulation"


def test_urban_heat_context_documents_equation_and_presentation():
    descriptor = simulation_dataset_descriptors()["urban_heat_simulation"]
    args = descriptor.validate({"bounds": (0.0, 0.0, 1.0, 1.0)})
    context = descriptor.create_context(args)
    manifest = context.manifest()

    assert manifest.identity.title == "Urban Heat Simulation"
    assert manifest.metadata.provider[0]["name"] == "DTCC Sim"
    assert manifest.metadata.source[1]["unknown"] == "temperature field T"
    assert manifest.metadata.source[1]["unit"] == "degC"
    assert manifest.metadata.collection_period.startswith("Computed on demand")
    assert "temperature_field" in manifest.metadata.data_types
    assert manifest.provenance.derived_from[0]["name"] == "city_volume_mesh"
    assert "steady diffusion/reaction" in manifest.provenance.processing_steps[4]
    assert manifest.presentation.headline == "Steady Urban Heat Field"
    assert manifest.presentation.legend["title"] == "Temperature field"
    assert manifest.presentation.view_hints["field"] == "temperature"
    assert manifest.presentation.warnings
    assert manifest.presentation.limitations
    assert manifest.request.parameters["wall_bc_type"] == "robin"
    assert manifest.request.parameters["open_bc_type"] == "dirichlet"


def test_urban_wind_context_documents_cfd_model_and_presentation():
    descriptor = simulation_dataset_descriptors()["urban_wind_simulation"]
    args = descriptor.validate(
        {
            "bounds": (0.0, 0.0, 1.0, 1.0),
            "equations": "stokes",
            "wind_dir_deg": 270.0,
            "inlet_profile": "log_law",
        }
    )
    context = descriptor.create_context(args)
    manifest = context.manifest()

    assert manifest.identity.title == "Urban Wind Simulation"
    assert manifest.metadata.provider[0]["name"] == "DTCC Sim"
    assert manifest.metadata.source[1]["unknowns"] == (
        "velocity u and kinematic pressure p/rho"
    )
    assert manifest.metadata.source[1]["units"]["pressure"] == "m^2/s^2"
    assert "velocity_field" in manifest.metadata.data_types
    assert "kinematic_pressure_field" in manifest.metadata.data_types
    assert manifest.provenance.derived_from[0]["name"] == "city_volume_mesh"
    assert "meteorological wind direction" in manifest.provenance.processing_steps[4]
    assert "convergence" in manifest.provenance.processing_steps[8]
    assert manifest.presentation.headline == "Urban Wind CFD Field"
    assert manifest.presentation.legend["title"] == "Wind fields"
    assert manifest.presentation.view_hints["default_color_attribute"] == "speed"
    assert manifest.presentation.view_hints["pressure_type"] == "kinematic"
    assert manifest.presentation.warnings
    assert manifest.presentation.limitations
    assert manifest.request.parameters["equations"] == "stokes"
    assert manifest.request.parameters["weather_aggregation"] == "nearest"
    assert manifest.request.parameters["divergence_tolerance"] == 5e-3
    assert manifest.request.parameters["inlet_profile"] == "log_law"


def test_air_quality_field_context_documents_reconstruction_and_lineage():
    descriptor = simulation_dataset_descriptors()["air_quality_field"]
    args = descriptor.validate(
        {
            "bounds": (0.0, 0.0, 1.0, 1.0),
            "phenomenon": "PM10",
            "lambda_smooth": 0.5,
            "data_weight": 200.0,
        }
    )
    context = descriptor.create_context(args)
    manifest = context.manifest()

    assert manifest.identity.title == "Air-Quality Reconstruction Field"
    assert manifest.metadata.provider[0]["name"] == "DTCC Sim"
    assert manifest.metadata.provider[2]["role"] == "source_provider"
    assert manifest.metadata.source[0]["name"] == "dtcc_core.datasets.air_quality"
    assert manifest.metadata.source[2]["unknown"] == (
        "scalar reconstructed concentration field u"
    )
    assert "station_observations" in manifest.metadata.data_types
    assert "reconstructed_concentration_field" in manifest.metadata.data_types
    assert manifest.provenance.derived_from[0]["name"] == "air_quality"
    assert "station coordinates" in manifest.provenance.processing_steps[2]
    assert "Tikhonov" in manifest.provenance.processing_steps[8]
    assert manifest.presentation.headline == "Derived Air-Quality Field"
    assert manifest.presentation.legend["title"] == "Reconstructed concentration"
    assert manifest.presentation.view_hints["upstream_observation_dataset"] == (
        "air_quality"
    )
    assert manifest.presentation.warnings
    assert manifest.presentation.limitations
    assert manifest.request.parameters["phenomenon"] == "PM10"
    assert manifest.request.parameters["lambda_smooth"] == 0.5
    assert manifest.request.parameters["data_weight"] == 200.0


def test_traffic_context_documents_assignment_model_and_lineage():
    descriptor = simulation_dataset_descriptors()["traffic_simulation"]
    args = descriptor.validate(
        {
            "bounds": (0.0, 0.0, 1.0, 1.0),
            "trips_per_employed": 1.5,
            "peak_hour_factor": 0.2,
            "bidirectional": False,
        }
    )
    context = descriptor.create_context(args)
    manifest = context.manifest()

    assert manifest.identity.title == "Static Traffic Assignment"
    assert manifest.metadata.provider[0]["name"] == "DTCC Sim"
    assert manifest.metadata.source[0]["name"] == "dtcc_core.datasets.roads"
    assert manifest.metadata.source[1]["required_statistics"] == [
        "population",
        "employment",
    ]
    assert "BPR travel time" in manifest.metadata.source[2]["cost_model"]
    assert "synthetic_od_matrix" in manifest.metadata.data_types
    assert "traffic_flow_attributes" in manifest.metadata.data_types
    assert manifest.provenance.derived_from[0]["name"] == "roads"
    assert manifest.provenance.derived_from[1]["name"] == "deso"
    assert "directed drivable graph" in manifest.provenance.processing_steps[2]
    assert "Frank-Wolfe" in manifest.provenance.processing_steps[6]
    assert manifest.presentation.headline == "Synthetic Peak-Hour Traffic"
    assert manifest.presentation.legend["title"] == "Traffic assignment attributes"
    assert manifest.presentation.view_hints["default_color_attribute"] == (
        "volume_capacity_ratio"
    )
    assert manifest.presentation.warnings
    assert manifest.presentation.limitations
    assert manifest.request.parameters["trips_per_employed"] == 1.5
    assert manifest.request.parameters["peak_hour_factor"] == 0.2
    assert manifest.request.parameters["bidirectional"] is False
    assert manifest.request.parameters["exclude_self_trips"] is True


def test_simulation_contract_audit_has_no_missing_findings():
    findings = audit_simulation_dataset_contracts()

    assert findings
    assert {finding.dataset for finding in findings} == set(SIMULATION_DATASET_NAMES)
    assert [finding for finding in findings if finding.status == "missing"] == []


def test_validation_cases_cover_major_simulation_families():
    cases = simulation_validation_cases()

    assert {case.family for case in cases} == {
        "urban_heat",
        "urban_wind",
        "air_quality_field",
        "traffic",
    }
    assert {case.dataset for case in cases} == set(SIMULATION_DATASET_NAMES)
    assert all(case.status in {"planned", "implemented"} for case in cases)
    assert all(case.test_reference for case in cases)


def test_qa_matrix_covers_registered_simulation_datasets():
    matrix = (REPO_ROOT / "docs" / "datasets" / "qa-matrix.md").read_text(
        encoding="utf-8"
    )

    for name in SIMULATION_DATASET_NAMES:
        assert f"`{name}`" in matrix
