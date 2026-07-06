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
