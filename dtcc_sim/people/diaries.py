"""Weighted weekday journey templates prepared from an authorized local survey."""

from hashlib import sha256
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

ARCHIVE = "rvu-diaries-2019.json"
DEPARTURE_BIN_SECONDS = 1800


class Journey(BaseModel):
    """One main journey with a half-hour departure bin and an activity purpose."""

    purpose: Literal[
        "home", "work", "study", "education", "escort", "shopping", "leisure"
    ]
    mode: Literal["walk", "cycle", "car", "transit"]
    departure_bin: int = Field(ge=0, lt=48)


class Diary(BaseModel):
    """A weighted home-to-home chain. An empty chain records a day at home."""

    weight: float = Field(gt=0, allow_inf_nan=False)
    journeys: list[Journey]

    @model_validator(mode="after")
    def validate_chain(self):
        if self.journeys and (
            self.journeys[0].purpose == "home" or self.journeys[-1].purpose != "home"
        ):
            raise ValueError("Diary journeys must start from home and finish at home")
        bins = [journey.departure_bin for journey in self.journeys]
        if bins != sorted(bins):
            raise ValueError("Diary departure bins must be chronological")
        return self


class Cohort(BaseModel):
    """Population age range for a set of comparable weekday templates."""

    age_min: int = Field(ge=6, le=84)
    age_max: int = Field(ge=6, le=84)
    diaries: list[Diary] = Field(min_length=1)


class DiaryFile(BaseModel):
    """Anonymous survey templates and their upstream reference."""

    reference_year: Literal[2019]
    day_type: Literal["weekday"]
    source_name: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    cohorts: list[Cohort] = Field(min_length=1)


class DiaryProfiles:
    """Sample a linked day using the survey weights within an age cohort."""

    def __init__(self, source):
        self.by_age = {}
        for cohort in source.cohorts:
            ages = range(cohort.age_min, cohort.age_max + 1)
            if not ages or self.by_age.keys() & set(ages):
                raise ValueError(
                    "Diary cohort age intervals must be nonempty and disjoint"
                )
            weights = np.asarray([diary.weight for diary in cohort.diaries])
            weights = weights / weights.max()
            entry = (cohort.diaries, weights / weights.sum())
            self.by_age.update(dict.fromkeys(ages, entry))

    def choose(self, age, stream):
        if age not in self.by_age:
            return None
        diaries, weights = self.by_age[age]
        return diaries[stream.choice(len(diaries), p=weights)].journeys


def load_diaries(root: Path):
    """Load the restricted template file without recording row-level data in a package."""
    path = root / ARCHIVE
    if not path.is_file():
        raise ValueError(
            f"Archived local travel diary templates are required at {path}"
        )
    payload = path.read_bytes()
    source = DiaryFile.model_validate_json(payload)
    return DiaryProfiles(source), {
        "name": source.source_name,
        "url": source.source_url,
        "vintage": source.reference_year,
        "archive": path.name,
        "sha256": sha256(payload).hexdigest(),
        "terms": "Authorized local use. Input template records are excluded from exports.",
        "selection": {"day_type": source.day_type, "cohorts": len(source.cohorts)},
    }
