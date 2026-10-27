"""Pin every Core dependency, including extras, to one tested revision."""

import argparse
import re
from pathlib import Path


def update_pin(path, sha):
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Core revision must be a full lowercase 40-character SHA")
    text = path.read_text(encoding="utf-8")
    pattern = (
        r"(dtcc-core(?:\[[^\]\n]+\])? @ git\+https://github\.com/"
        r"dtcc-platform/dtcc-core\.git@)(?:develop|[0-9a-f]{40})"
    )
    updated, count = re.subn(pattern, lambda match: match[1] + sha, text)
    if count == 0:
        raise ValueError("No supported dtcc-core dependencies found")
    path.write_text(updated, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--project", type=Path, default=Path("pyproject.toml"))
    args = parser.parse_args()
    update_pin(args.project, args.sha)


if __name__ == "__main__":
    main()
