from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-expensive",
        action="store_true",
        default=False,
        help="Run expensive/manual-scale simulation tests.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-expensive"):
        return

    skip_expensive = pytest.mark.skip(
        reason="expensive simulation tests require --run-expensive"
    )
    for item in items:
        if "expensive" in item.keywords:
            item.add_marker(skip_expensive)
