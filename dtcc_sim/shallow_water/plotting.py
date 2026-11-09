"""Shallow-water display defaults for native Core face maps."""

import numpy as np


def _plot_depth(mesh, time, *, ax=None, show=True, vmax=None, theme="light"):
    """Delegate the selected snapshot's metric depth map to Core."""
    markers = np.asarray(mesh.markers)
    if markers.size and markers.shape != (len(mesh.faces),):
        raise ValueError("Markers must have one entry per face")
    context = markers >= 0 if markers.size else None
    return mesh.plot(
        projection="xy", field="water_depth", context_mask=context,
        title=f"Water depth · t = {time:g} s", coordinate_unit="m", scale_bar=True,
        vmin=0., vmax=vmax, cmap="Blues", ax=ax, show=show, theme=theme,
    )
