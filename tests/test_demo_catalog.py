"""The catalog covers short, linear public API examples."""

import ast
from pathlib import Path
import re

import pytest


DEMO_DIR = Path(__file__).resolve().parents[1] / "demos"
DEMOS = sorted(DEMO_DIR.glob("*.py"))


def test_catalog_covers_each_demo_once():
    catalog = (DEMO_DIR / "README.md").read_text()
    entries = re.findall(r"\| \[([^\]]+\.py)\]\(([^)]+)\)", catalog)
    assert sorted(entries) == [(path.name, path.name) for path in DEMOS]
    datasets, solvers = catalog.split("## Solver examples", 1)
    assert set(re.findall(r"\]\(([^)]+\.py)\)", datasets)) == {
        path.name for path in DEMOS if path.stem.startswith("dataset_")
    }
    assert set(re.findall(r"\]\(([^)]+\.py)\)", solvers)) == {
        path.name for path in DEMOS if not path.stem.startswith("dataset_")
    }


@pytest.mark.parametrize("path", DEMOS, ids=lambda path: path.stem)
def test_demo_structure(path):
    source = path.read_text()
    tree = ast.parse(source)
    assert ast.get_docstring(tree)
    assert len(source.splitlines()) <= 40
    assert not any(isinstance(node, (
        ast.AsyncFunctionDef, ast.ClassDef, ast.If,
        ast.For, ast.While, ast.Try,
    )) for node in ast.walk(tree))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            assert len(node.body) == 1 and isinstance(node.body[0], ast.Return)
    assert "argparse" not in source and "__main__" not in source
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(isinstance(call.func, ast.Attribute) and call.func.attr == "info"
               for call in calls)
    assert any(isinstance(call.func, ast.Attribute) and call.func.attr == "plot"
               for call in calls)
    output, = [keyword.value for call in calls for keyword in call.keywords
               if keyword.arg == "output_dir"]
    if isinstance(output, ast.Name):
        output, = [node.value for node in tree.body if isinstance(node, ast.Assign)
                   and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == output.id]
    assert ast.literal_eval(output) == f"output/{path.stem}"


def test_dataset_wind_uses_same_scenario_as_direct_solver():
    def keywords(path, function):
        tree = ast.parse((DEMO_DIR / path).read_text())
        call, = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and ((isinstance(node.func, ast.Name) and node.func.id == function)
                      or (isinstance(node.func, ast.Attribute) and node.func.attr == function))]
        return {item.arg: ast.literal_eval(item.value) for item in call.keywords
                if item.arg != "bounds"}

    direct = keywords("urban_wind_simulation.py", "NavierStokesParameters")
    dataset = keywords("dataset_urban_wind.py", "urban_wind_simulation")
    assert dataset.pop("equations") == "navier_stokes"
    assert dataset.pop("output_dir") == "output/dataset_urban_wind"
    assert dataset == direct
