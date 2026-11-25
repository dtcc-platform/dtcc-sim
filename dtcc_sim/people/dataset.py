"""Registered synthetic people dataset."""

from copy import deepcopy
from typing import ClassVar, Literal

from dtcc_core.datasets import Dataset, DatasetBaseArgs
from pydantic import Field, model_validator

VALIDATION_REFERENCE = {
    "relationship": "model_evaluation",
    "version": "2019-v1",
    "model_version": "0.1.0",
    "url": "https://github.com/dtcc-platform/dtcc-sim/blob/develop/docs/people-validation.md",
    "evidence": {
        "gapsim": {
            "doi": "10.5281/zenodo.13882078",
            "reference_year": 2019,
            "area": ["Johanneberg", "Askim", "Hammarkullen"],
            "tested_outputs": [
                "resident demography",
                "household composition",
                "home placement",
                "adult daily schedules",
                "main mode and purpose",
                "walking route distributions",
            ],
            "field_observation": False,
        },
        "independent_pedestrian_observations": {
            "status": "unscored",
            "reason": "Comparable 2019 screenline intervals and audited geometry were unavailable.",
        },
    },
}
SCHEDULED_VALIDATION_REFERENCE = {
    **VALIDATION_REFERENCE,
    "version": "2019-v1-scheduled",
    "url": "https://github.com/dtcc-platform/dtcc-sim/blob/develop/docs/people-validation.md#scheduled-transit-sensitivity",
    "applies_to": {"transit_service_date": "2019-10-15", "survey_profile": "published"},
}
JOHANNEBERG_VALIDATION_REFERENCE = {
    **VALIDATION_REFERENCE,
    "version": "2019-v2-johanneberg",
    "model_version": "0.2.1",
    "url": "https://github.com/dtcc-platform/dtcc-sim/blob/develop/docs/people-validation.md#johanneberg-model-021",
    "applies_to": {
        "region": "johanneberg",
        "transit_service_date": None,
        "survey_profile": "published",
        "seeds": [17, 42, 83],
        "nearest_baseline_seed": 42,
    },
    "evidence": {
        **VALIDATION_REFERENCE["evidence"],
        "gapsim": {
            **VALIDATION_REFERENCE["evidence"]["gapsim"],
            "area": ["Johanneberg"],
        },
    },
}


class PeopleArgs(DatasetBaseArgs):
    """Swedish synthetic people scenario selection."""

    reference_year: Literal[2019] = Field(
        2019, description="Population and survey reference year"
    )
    day_type: Literal["weekday"] = Field(
        "weekday", description="Representative term-time weekday"
    )
    seed: int = Field(42, ge=0, le=2**32 - 1, description="Regional synthesis seed")
    region: str | None = Field(None, description="Named archived regional scenario")
    closed_walk_edge: str | None = Field(
        None, description="OSM way:segment excluded for a paired walking intervention"
    )
    baseline: bool = Field(
        False, description="Nearest-anchor and shortest-distance comparison baseline"
    )
    transit_service_date: Literal["2019-10-15"] | None = Field(
        None, description="Use archived scheduled public transport on this local date"
    )
    survey_profile: Literal["published", "local_diaries"] = Field(
        "published",
        description="Public RVU tables or an authorized local weekday diary template archive",
    )
    profile: Literal["population", "walking_preview"] = Field(
        "population",
        description="Regional synthetic population or small walking playback sample",
    )

    @model_validator(mode="after")
    def validate_population_options(self):
        if self.profile != "population" and (
            self.transit_service_date
            or self.survey_profile != "published"
            or self.baseline
            or self.closed_walk_edge is not None
        ):
            raise ValueError(
                "Scheduled transit, local diaries, and scenario comparisons require the population profile"
            )
        return self


class PeopleDataset(Dataset):
    """Synthetic residents and timed pedestrian paths in Sweden."""

    name = "people"
    version = "0.2.1"
    title = "People"
    description = (
        "Synthetic people and inferred trips on a representative Swedish weekday."
    )
    ArgsModel = PeopleArgs
    data_category = "simulation"
    result_kind = "people_simulation"
    python_return_type = "dtcc_core.model.PeopleSimulation"
    default_crs = "EPSG:3006"
    data_types: ClassVar = ["synthetic_population", "timed_trajectories"]
    timeout_hint = 600
    export_capabilities: ClassVar = [
        {
            "product": None,
            "canonical": True,
            "web_formats": ["geojson"],
            "constraints": {},
            "options": {},
            "exclusions": [],
        }
    ]
    presentation_summary = (
        "Synthetic people on a representative weekday in Europe/Stockholm."
    )
    presentation_warnings: ClassVar = [
        "People and trips are synthetic. Pedestrian flow accuracy requires independent field counts."
    ]
    presentation_limitations: ClassVar = [
        "Current road mapping and 2019 population data have different vintages.",
        "Building access points and daily schedules are inferred.",
    ]

    def build(self, args):
        from .scenario import build

        return build(self.parse_bounds(args.bounds), args)

    def _context_for_result(self, result, args):
        context = self.create_context(args)
        context.provenance.sources = result.attributes.pop("source_provenance")
        context.metadata.source = context.provenance.sources
        context.provenance.generated_by = {
            "package": "dtcc-sim",
            "model_version": self.version,
        }
        context.provenance.processing_steps += [result.attributes["diagnostics"]]
        context.presentation.annotations.append(
            {
                "relationship": "methods",
                "url": "https://github.com/dtcc-platform/dtcc-sim/blob/develop/docs/people.md",
            }
        )
        if args.transit_service_date:
            context.presentation.limitations.append(
                "Transit follows the planned timetable. Ride geometry follows mapped tram rails and roads where found, otherwise straight lines between stop areas."
            )
        if args.survey_profile == "local_diaries":
            context.presentation.limitations.append(
                "The local diary template profile has no GAPSIM or independent pedestrian evaluation."
            )
        elif args.profile == "population":
            if (
                result.attributes.get("region") == "johanneberg"
                and not args.transit_service_date
                and not args.closed_walk_edge
            ):
                context.presentation.annotations.append(
                    deepcopy(JOHANNEBERG_VALIDATION_REFERENCE)
                )
            else:
                reference = (
                    SCHEDULED_VALIDATION_REFERENCE
                    if args.transit_service_date
                    else VALIDATION_REFERENCE
                )
                context.presentation.annotations.append(
                    {**deepcopy(reference), "relationship": "prior_model_evaluation"}
                )
                context.presentation.limitations.append(
                    f"The linked GAPSIM evaluation covers model {reference['model_version']}."
                )
        return context
