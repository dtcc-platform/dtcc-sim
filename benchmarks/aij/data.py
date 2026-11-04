"""Readers for the AIJ benchmark workbooks.

The workbooks are distributed by the Architectural Institute of Japan at
https://www.aij.or.jp/jpn/publish/cfdguide/index_e.htm and are not part of
this repository; place them in ``benchmarks/data/aij/``. Reading ``.xls``
files needs ``xlrd``.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "aij"

_COLUMNS = {"x": "x", "y": "y", "z": "z", "u": "U", "v": "V", "w": "W", "k": "k"}


@dataclass
class Section:
    """Measured mean velocity (m/s) and turbulent kinetic energy (m²/s²) at points (m)."""

    name: str
    number: np.ndarray
    points: np.ndarray
    velocity: np.ndarray
    k: np.ndarray


@dataclass
class Measurements:
    """Approach flow profile and measured sections of one AIJ case."""

    inflow_z: np.ndarray
    inflow_speed: np.ndarray
    inflow_k: np.ndarray
    sections: dict = field(default_factory=dict)


def _scale(header, length):
    text = header.replace(" ", "")
    if "[mm]" in text:
        return 1e-3
    if "/b" in text:
        return length
    if "[m]" in text or "(m)" in text:
        return 1.
    return None


def _table(sheet, length):
    """Header row and numeric rows of a sheet as {column: values}."""
    for row in range(sheet.nrows):
        cells = [str(sheet.cell_value(row, c)).strip() for c in range(sheet.ncols)]
        if any(re.match(r"^U\s*[\[(]", cell) for cell in cells):
            break
    else:
        raise ValueError(f"no velocity header in sheet {sheet.name!r}")
    columns, scales = {}, {}
    for c, cell in enumerate(cells):
        match = re.match(r"^([xyzuvwk])\b", cell, flags=re.IGNORECASE)
        key = _COLUMNS.get(match.group(1).lower()) if match else None
        if key and key not in columns and (key in "xyzk" or cell[0].isupper()):
            columns[key], scales[key] = c, _scale(cell, length)
    if "x" in scales and scales["x"] is None:
        raise ValueError(f"unknown coordinate unit in sheet {sheet.name!r}")
    for key in "yz":
        if key in scales and scales[key] is None:
            scales[key] = scales.get("x")
    number_column = 0
    values = {key: [] for key in columns}
    values["number"] = []
    for r in range(row + 1, sheet.nrows):
        raw = {key: sheet.cell_value(r, c) for key, c in columns.items()}
        if not all(isinstance(v, float) for v in raw.values()):
            continue
        for key, value in raw.items():
            values[key].append(value * (scales.get(key) or 1.) if key in "xyz" else value)
        number = sheet.cell_value(r, number_column)
        values["number"].append(int(number) if isinstance(number, float) else len(values["number"]) + 1)
    return {key: np.asarray(v) for key, v in values.items()}


def _section_height(name, length):
    match = re.search(r"z\s*=\s*([0-9.]+)\s*b", name)
    return float(match.group(1)) * length if match else None


def read_workbook(path, length):
    """Measurements from an AIJ workbook; ``length`` is the reference size b (m)
    for coordinates given as multiples of b.

    Horizontal sections named ``z=…b`` take their height from the sheet name.
    """
    import xlrd

    book = xlrd.open_workbook(str(path))
    inflow = _table(book.sheet_by_name("Inflow"), length)
    order = np.argsort(inflow["z"])
    measurements = Measurements(inflow["z"][order], inflow["U"][order], inflow["k"][order])
    for sheet in book.sheets():
        if not sheet.name.startswith("Results"):
            continue
        table = _table(sheet, length)
        points = np.column_stack([table["x"], table["y"], table["z"]])
        height = _section_height(sheet.name, length)
        if height is not None:
            points[:, 2] = height
        name = "vertical" if "Ver" in sheet.name else "horizontal"
        if height is not None:
            name += f" z={height / length:g}b"
        measurements.sections[name] = Section(
            name, table["number"], points, np.column_stack([table["U"], table["V"], table["W"]]), table["k"])
    return measurements
