"""Offline QA helpers for dtcc-sim simulation datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SimulationQAStatus = Literal["present", "missing", "planned", "implemented"]

SIMULATION_DATASET_NAMES: tuple[str, ...] = (
    "urban_heat_simulation",
    "urban_wind_simulation",
    "air_quality_field",
    "traffic_simulation",
)


@dataclass(frozen=True)
class SimulationDatasetQAFinding:
    """One static QA finding for a registered simulation dataset."""

    dataset: str
    field: str
    status: SimulationQAStatus
    message: str


@dataclass(frozen=True)
class SimulationValidationCase:
    """Documented validation case for one simulation family."""

    family: str
    dataset: str
    status: SimulationQAStatus
    scope: str
    test_reference: str
    notes: str


VALIDATION_CASES: tuple[SimulationValidationCase, ...] = (
    SimulationValidationCase(
        family="urban_heat",
        dataset="urban_heat_simulation",
        status="implemented",
        scope="field attachment and XDMF serialization contract",
        test_reference="tests/test_urban_heat_dataset_v2.py",
        notes="Full solver validation remains a slow/FEniCSx environment concern.",
    ),
    SimulationValidationCase(
        family="urban_wind",
        dataset="urban_wind_simulation",
        status="implemented",
        scope="wind direction/profile math plus a small IPCS smoke test",
        test_reference="tests/test_urban_wind.py",
        notes="The small IPCS case covers numerical sanity; larger CFD validation is planned separately.",
    ),
    SimulationValidationCase(
        family="air_quality_field",
        dataset="air_quality_field",
        status="implemented",
        scope="sensor-to-field plumbing and XDMF serialization contract",
        test_reference="tests/test_air_quality_dataset_v2.py",
        notes="Live SMHI/provider drift is intentionally outside default CI.",
    ),
    SimulationValidationCase(
        family="traffic",
        dataset="traffic_simulation",
        status="implemented",
        scope="deterministic road assignment flow/capacity attributes",
        test_reference="tests/test_traffic.py",
        notes="Uses small synthetic roads/zones and no provider network access.",
    ),
)


def simulation_validation_cases() -> tuple[SimulationValidationCase, ...]:
    """Return documented validation status for major simulation families."""
    return VALIDATION_CASES


def simulation_dataset_descriptors():
    """Return registered dtcc-sim dataset descriptors by public dataset name."""
    import dtcc_core.datasets as datasets
    import dtcc_sim.datasets  # noqa: F401  Ensure registration side effects.

    descriptors = {}
    for name in SIMULATION_DATASET_NAMES:
        descriptor = getattr(datasets, name, None)
        if descriptor is None:
            raise RuntimeError(f"dtcc-sim dataset is not registered: {name}")
        descriptors[name] = descriptor
    return descriptors


def audit_simulation_dataset_contracts() -> tuple[SimulationDatasetQAFinding, ...]:
    """Audit registered simulation descriptors without running simulations."""
    findings: list[SimulationDatasetQAFinding] = []
    for name, descriptor in simulation_dataset_descriptors().items():
        description = descriptor.describe()
        _record_required_field(findings, name, description, "description")
        _record_required_field(findings, name, description, "result_kind")
        _record_required_field(findings, name, description, "python_return_type")
        _record_required_field(findings, name, description, "timeout_hint")
        if description.get("data_category") == "simulation":
            findings.append(
                SimulationDatasetQAFinding(
                    dataset=name,
                    field="data_category",
                    status="present",
                    message="data_category is simulation.",
                )
            )
        else:
            findings.append(
                SimulationDatasetQAFinding(
                    dataset=name,
                    field="data_category",
                    status="missing",
                    message="data_category must be simulation.",
                )
            )

        args = descriptor.validate({"bounds": (0.0, 0.0, 1.0, 1.0)})
        context = descriptor.create_context(args)
        if context.request.dataset_name == name and context.request.bounds:
            findings.append(
                SimulationDatasetQAFinding(
                    dataset=name,
                    field="context",
                    status="present",
                    message="descriptor can create Dataset v2 context offline.",
                )
            )
        else:
            findings.append(
                SimulationDatasetQAFinding(
                    dataset=name,
                    field="context",
                    status="missing",
                    message="descriptor context is missing dataset name or bounds.",
                )
            )
    return tuple(findings)


def _record_required_field(
    findings: list[SimulationDatasetQAFinding],
    dataset: str,
    description: dict,
    field: str,
) -> None:
    value = description.get(field)
    if value:
        findings.append(
            SimulationDatasetQAFinding(
                dataset=dataset,
                field=field,
                status="present",
                message=f"{field} is populated.",
            )
        )
    else:
        findings.append(
            SimulationDatasetQAFinding(
                dataset=dataset,
                field=field,
                status="missing",
                message=f"{field} is required for simulation dataset QA.",
            )
        )
