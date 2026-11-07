"""Plan-view presentation of conservative shallow-water face fields."""

from numbers import Real

import numpy as np


def _plot_depth(mesh, time, *, ax=None, show=True, vmax=None, theme="light"):
    """Fill actual ground triangles and outline the union of building faces."""
    from dtcc_core.plotting.style import (
        apply_dtcc_style, get_theme, require_matplotlib, style_colorbar,
    )
    if vmax is not None and (isinstance(vmax, bool) or not isinstance(vmax, Real)
                             or not np.isfinite(vmax) or vmax <= 0):
        raise ValueError("vmax must be finite and positive")
    if ax is not None and getattr(ax, "name", None) != "rectilinear":
        raise ValueError("Water-depth maps require a Matplotlib 2D axes")
    depth = next(field for field in mesh.fields if field.name == "water_depth").values[:, 0]
    xy = np.asarray(mesh.vertices)[:, :2]
    faces = np.asarray(mesh.faces)
    markers = np.asarray(mesh.markers)
    building = markers >= 0 if markers.size else np.zeros(len(faces), dtype=bool)
    ground = ~building
    if not ground.any():
        raise ValueError("Water-depth maps require at least one ground face")
    limit = float(vmax) if vmax is not None else max(float(depth[ground].max()), 1e-12)
    plt = require_matplotlib("shallow-water depth maps")
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    colors = get_theme(theme)
    cmap = plt.get_cmap("Blues")
    water = PolyCollection(xy[faces[ground]], array=depth[ground], cmap=cmap,
                           norm=Normalize(vmin=0., vmax=limit), edgecolors="none")
    water.set_gid("water_depth")
    ax.add_collection(water)
    handles = [Patch(facecolor=cmap(0.), label="Dry ground")]
    if building.any():
        footprints = PolyCollection(xy[faces[building]], facecolors=colors["muted"],
                                    edgecolors="none")
        footprints.set_gid("buildings")
        ax.add_collection(footprints)
        edges = np.sort(np.concatenate([faces[building][:, pair]
                                         for pair in ([0, 1], [1, 2], [2, 0])]), axis=1)
        boundary, counts = np.unique(edges, axis=0, return_counts=True)
        outlines = LineCollection(xy[boundary[counts == 1]], colors=colors["text"],
                                   linewidths=.7)
        outlines.set_gid("building_outlines")
        ax.add_collection(outlines)
        handles.append(Patch(facecolor=colors["muted"], edgecolor=colors["text"], label="Buildings"))
    crs = mesh.transform.srs
    projected = crs == "EPSG:3006"
    apply_dtcc_style(ax, theme=theme, equal_aspect=True,
                     title=f"Water depth · t = {time:g} s",
                     xlabel="Easting [m]" if projected else "X [m]",
                     ylabel="Northing [m]" if projected else "Y [m]")
    ax.set_gid("water_depth_map")
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = float((hi - lo).max())
    padding = .02 * span
    ax.set(xlim=(lo[0] - padding, hi[0] + padding),
           ylim=(lo[1] - padding, hi[1] + padding))
    for axis in (ax.xaxis, ax.yaxis):
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
        axis.set_major_locator(MaxNLocator(nbins=5))
    legend = ax.legend(handles=handles, loc="upper right", fontsize=8,
                       facecolor=colors["panel"], edgecolor=colors["grid"], labelcolor=colors["text"])
    legend.set_zorder(5)
    if depth[ground].max() > 0 or vmax is not None:
        style_colorbar(ax.figure.colorbar(water, ax=ax, pad=.03, fraction=.045,
                                          label="Water depth [m]"), theme=theme)
    else:
        ax.text(.5, .96, "All ground cells dry", transform=ax.transAxes,
                ha="center", va="top", color=colors["text"])
    target = span / 5
    magnitude = 10. ** np.floor(np.log10(target))
    scale = (5 if target / magnitude >= 5 else 2 if target / magnitude >= 2 else 1) * magnitude
    bar = AnchoredSizeBar(ax.transData, scale, f"{scale:g} m", "lower left",
                          pad=.3, borderpad=.5, frameon=True, color=colors["text"],
                          size_vertical=span * .003, fontproperties={"size": 8})
    bar.patch.set(facecolor=colors["panel"], edgecolor=colors["grid"], alpha=.9)
    bar.set_gid("water_depth_scale")
    ax.add_artist(bar)
    if crs:
        ax.text(0., 1.01, crs, transform=ax.transAxes, fontsize=8,
                color=colors["muted"], ha="left", va="bottom")
    if show:
        plt.show()
    return ax
