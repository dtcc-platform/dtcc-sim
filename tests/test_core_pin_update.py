"""Check the dependency-pin updater without network or package installation."""

import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/update_core_pin.py"
CORE = "dtcc-core @ git+https://github.com/dtcc-platform/dtcc-core.git@"
SHA = "a" * 40


@pytest.mark.parametrize("previous", ["develop", "b" * 40])
def test_base_and_volume_extra_pin_together(tmp_path, previous):
    project = tmp_path / "pyproject.toml"
    volume = CORE.replace("dtcc-core @", "dtcc-core[volume] @")
    project.write_text(f"{CORE}{previous}\n{volume}{previous}\nother-package\n")
    subprocess.run(
        [sys.executable, str(SCRIPT), "--project", str(project), "--sha", SHA],
        check=True,
        capture_output=True,
    )
    assert project.read_text() == f"{CORE}{SHA}\n{volume}{SHA}\nother-package\n"


@pytest.mark.parametrize(
    "sha,contents", [("invalid", CORE + "develop"), (SHA, "other-package")]
)
def test_invalid_update_does_not_modify_project(tmp_path, sha, contents):
    project = tmp_path / "pyproject.toml"
    project.write_text(contents)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--project", str(project), "--sha", sha],
        capture_output=True,
    )
    assert result.returncode != 0
    assert project.read_text() == contents
