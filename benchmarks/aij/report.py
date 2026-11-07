"""Figures and a metric summary from AIJ benchmark results.

    python -m benchmarks.aij.report output/benchmarks/aij

Reads every ``case-*/*/results.json`` below the directory and writes
``summary.md`` and one PNG per case and figure next to the results.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .analysis import log_law

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORDER = {"coarse": 0, "medium": 1, "fine": 2}


def load(directory):
    runs = {}
    for path in sorted(Path(directory).glob("case-*/*/results.json")):
        result = json.loads(path.read_text())
        runs.setdefault(result["case"], []).append(result)
    for results in runs.values():
        results.sort(key=lambda r: (r["turbulence_model"] != "mixing_length", ORDER.get(r["resolution"], 9)))
    return runs


def label(result):
    return f"{result['turbulence_model'].replace('_', ' ')}, {result['resolution']} ({result['mesh']['cells'] // 1000}k cells)"


def summary(runs):
    lines = ["| Case | Model | Resolution | Cells | Newton | Time (s) | Section | Speed hit rate | FAC2 | FB | R |",
             "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
    for case, results in sorted(runs.items()):
        for r in results:
            rows = [(name, s["metrics"]["speed"]) for name, s in r["sections"].items()] + [("all", r["metrics"]["speed"])]
            for name, m in rows:
                lines.append(
                    f"| {case} | {r['turbulence_model']} | {r['resolution']} | {r['mesh']['cells']} | "
                    f"{r['solver']['nonlinear_iterations']} | {r['solver']['seconds']:.0f} | {name} | "
                    f"{m['hit_rate']:.2f} | {m['fac2']:.2f} | {m['fractional_bias']:+.2f} | {m['correlation']:.2f} |")
    lines += ["", "Change of the computed |U| / U_H at the measurement points between resolutions:", "",
              "| Case | Model | Resolutions | Mean change | Largest change |", "| --- | --- | --- | ---: | ---: |"]
    for case, results in sorted(runs.items()):
        for model in sorted({r["turbulence_model"] for r in results}):
            levels = [r for r in results if r["turbulence_model"] == model]
            for a, b in zip(levels[:-1], levels[1:]):
                change = np.abs(speeds(b) - speeds(a))
                lines.append(f"| {case} | {model} | {a['resolution']} → {b['resolution']} | "
                             f"{np.nanmean(change):.3f} | {np.nanmax(change):.3f} |")
    return "\n".join(lines) + "\n"


def speeds(result, key="predicted"):
    return np.concatenate([np.linalg.norm(s[key], axis=1) for s in result["sections"].values()])


def style(ax):
    ax.grid(color=GRID, linewidth=.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def stations(section, axis):
    points = np.asarray(section["points_m"])
    return sorted({round(v, 6) for v in points[:, axis]})


def profiles(results, name, along, station_axis, length, scale_label, path, station_length):
    """Streamwise velocity along ``along`` at each station of one section."""
    import matplotlib.pyplot as plt

    first = results[0]["sections"][name]
    xs = stations(first, station_axis)
    fig, axes = plt.subplots(1, len(xs), figsize=(1.6 * len(xs) + 1, 3.6), sharey=True, squeeze=False)
    for ax, x in zip(axes[0], xs):
        style(ax)
        for i, r in enumerate(results):
            section = r["sections"][name]
            points = np.asarray(section["points_m"])
            at = np.isclose(points[:, station_axis], x)
            order = np.argsort(points[at, along])
            coordinate = points[at, along][order] / length
            if i == 0:
                ax.plot(np.asarray(section["observed"])[at, 0][order], coordinate, "o", ms=4.5,
                        color=INK, label="wind tunnel", zorder=3)
            ax.plot(np.asarray(section["predicted"])[at, 0][order], coordinate, "-", lw=1.5,
                    color=SERIES[i % len(SERIES)], label=label(r))
        ax.axvline(0, color=MUTED, lw=.6)
        ax.set_title(f"x/b = {x / station_length:g}", fontsize=8, color=INK)
        ax.set_xlabel("U / U_H", fontsize=8, color=MUTED)
    axes[0, 0].set_ylabel(scale_label, fontsize=8, color=MUTED)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=len(results) + 1,
               fontsize=8, frameon=False)
    fig.suptitle(f"Case {results[0]['case']}: {name}", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def scatter(results, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(3.3 * len(results), 3.9), squeeze=False)
    fig.suptitle(f"Case {results[0]['case']}: {results[0]['title']}, computed against measured speed",
                 fontsize=10, color=INK)
    for ax, r in zip(axes[0], results):
        style(ax)
        observed = np.concatenate([np.linalg.norm(s["observed"], axis=1) for s in r["sections"].values()])
        predicted = np.concatenate([np.linalg.norm(s["predicted"], axis=1) for s in r["sections"].values()])
        top = 1.1 * np.nanmax([observed.max(), np.nanmax(predicted)])
        line = np.array([0, top])
        ax.fill_between(line, .75 * line, 1.25 * line, color=GRID, lw=0)
        ax.plot(line, line, color=MUTED, lw=.8)
        ax.plot(observed, predicted, "o", ms=3.5, color=SERIES[0], alpha=.8)
        m = r["metrics"]["speed"]
        ax.set_title(f"{label(r)}\nhit rate {m['hit_rate']:.2f}, FAC2 {m['fac2']:.2f}", fontsize=8, color=INK)
        ax.set(xlim=(0, top), ylim=(0, top), aspect="equal")
        ax.set_xlabel("measured |U| / U_H", fontsize=8, color=MUTED)
        ax.set_ylabel("computed |U| / U_H", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def inflow(result, path):
    import matplotlib.pyplot as plt

    data = result["inflow"]
    z, speed = np.asarray(data["observed"]["z_m"]), np.asarray(data["observed"]["speed"])
    H = result["building_m"]["height"]
    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    style(ax)
    ax.plot(speed, z / H, "o", ms=4.5, color=INK, label="wind tunnel")
    fine = np.linspace(0, result["domain_m"][2], 200)
    ax.plot(log_law(fine, data["friction_velocity"], data["z0_m"]), fine / H, "-", lw=1.5, color=SERIES[0],
            label=f"log law, u* = {data['friction_velocity']:.3f} m/s, z0 = {1e3 * data['z0_m']:.3f} mm")
    ax.axhline(1, color=MUTED, lw=.6)
    ax.set_xlim(.8 * speed.min(), 1.05 * speed.max())
    ax.set_xlabel("U (m/s)", fontsize=8, color=MUTED)
    ax.set_ylabel("z / H", fontsize=8, color=MUTED)
    ax.set_title(f"Case {result['case']}: approach flow", fontsize=10, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path, nargs="?", default=Path("output/benchmarks/aij"))
    args = parser.parse_args()
    runs = load(args.directory)
    if not runs:
        raise SystemExit(f"no results below {args.directory}")
    (args.directory / "summary.md").write_text(summary(runs))
    for case, results in runs.items():
        out = args.directory / f"case-{case}"
        length = results[0]["building_m"]["depth"]
        H = results[0]["building_m"]["height"]
        inflow(results[0], out / "inflow.png")
        for i, name in enumerate(results[0]["sections"]):
            if name.startswith("vertical"):
                profiles(results, name, 2, 0, H, "z / H", out / f"section-{i}-vertical.png", length)
            else:
                profiles(results, name, 1, 0, length, "y / b", out / f"section-{i}-horizontal.png", length)
        scatter(results, out / "scatter.png")
    print((args.directory / "summary.md").read_text())


if __name__ == "__main__":
    main()
