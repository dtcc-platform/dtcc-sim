"""Weight support, normalization and construction-specific regression checks."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mpi4py", reason="Partition weights require the Helmholtz runtime")
pytest.importorskip("dolfinx", reason="Partition weights require the Helmholtz runtime")

from dtcc_sim.helmholtz._backend.params import Params, bind_parameters
from dtcc_sim.helmholtz._backend.pou import build_partition_of_unity


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_geometric_hats_and_zero_weight_fallback(dimension):
    # Each one-node cell fixes a prescribed barycentre for this geometry fixture.
    barycentres = np.array([[0.0], [1.5], [0.5], [2.0]])
    coordinates = np.array([[0.5], [1.0], [1.5], [0.0], [10.0]])
    if dimension > 1:
        barycentres = np.column_stack(
            [barycentres] + [np.array([0.0, 2.0, 0.0, 2.0])] * (dimension - 1)
        )
        coordinates = np.column_stack(
            [coordinates] + [np.array([1.0, 1.0, 1.0, 0.0, 10.0])] * (dimension - 1)
        )
    mesh = SimpleNamespace(
        geometry=SimpleNamespace(
            dim=dimension, x=barycentres, dofmap=np.arange(4).reshape(-1, 1)
        ),
        topology=SimpleNamespace(
            dim=dimension, index_map=lambda _: SimpleNamespace(size_local=4)
        ),
    )
    space = SimpleNamespace(tabulate_dof_coordinates=lambda: coordinates)
    decomposition = {
        "cells_olap": [np.array([0, 1]), np.array([2, 3]), np.array([], dtype=int)],
        "dofs_olap": [np.arange(4), np.arange(4), np.array([], dtype=int)],
    }
    with bind_parameters(Params(pou_mode="geometric_hats", verbose=False)):
        result = build_partition_of_unity(mesh, space, decomposition=decomposition)
    np.testing.assert_allclose(result.chi_full[0], [1, 0.5, 0, 0.5, 0], atol=1e-15)
    np.testing.assert_allclose(result.chi_full[1], [0, 0.5, 1, 0.5, 0], atol=1e-15)
    np.testing.assert_array_equal(result.chi_full[2], np.zeros(5))
    np.testing.assert_array_equal(result.multiplicity, [2, 2, 2, 2, 0])
    assert result.ndofs_in_union == 4


@pytest.mark.parametrize(
    "mode", ["core_overlap_ramp", "structured_hats", "from_partition_data"]
)
def test_ramp_and_internal_aliases_keep_core_layer_weights(mode):
    space = SimpleNamespace(
        tabulate_dof_coordinates=lambda: np.arange(5).reshape(-1, 1),
        dofmap=SimpleNamespace(cell_dofs=lambda c: np.array([c, c + 1])),
    )
    decomposition = {
        "cells_core": [np.array([0]), np.array([3])],
        "cells_olap": [np.array([0, 1, 2]), np.array([1, 2, 3])],
        "cell_adjacency": [
            np.array([1]),
            np.array([0, 2]),
            np.array([1, 3]),
            np.array([2]),
        ],
        "dofs_core": [np.array([0, 1]), np.array([3, 4])],
        "dofs_olap": [np.arange(4), np.arange(1, 5)],
        "overlap_layers": np.array([2]),
    }
    with bind_parameters(Params(pou_mode=mode, verbose=False)):
        result = build_partition_of_unity(None, space, decomposition=decomposition)
    np.testing.assert_allclose(result.chi_full[0], [1, 0.75, 0.5, 0.25, 0], atol=1e-15)
    np.testing.assert_allclose(np.sum(result.chi_full, axis=0), np.ones(5), atol=1e-15)
