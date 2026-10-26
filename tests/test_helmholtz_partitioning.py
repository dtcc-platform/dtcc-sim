"""Partitioner seed forwarding and repeatability checks in the numerical runtime."""

import numpy as np
import pytest

from dtcc_sim.helmholtz import HelmholtzParameters


@pytest.mark.parametrize("seed", [-2, True, 1.5, 2147483648])
def test_invalid_seed(seed):
    with pytest.raises(ValueError, match="metis_seed"):
        HelmholtzParameters(metis_seed=seed)


@pytest.mark.parametrize("seed", [-1, 0, 13])
def test_seed_reaches_pymetis_without_changing_numpy_rng(monkeypatch, seed):
    pymetis = pytest.importorskip("pymetis")
    pytest.importorskip("dolfinx")
    from dtcc_sim.helmholtz._backend.decomposition import metis_partition_cells

    real = pymetis.part_graph
    calls = []

    def spy(nparts, *, adjacency, options):
        calls.append(options.seed)
        return real(nparts, adjacency=adjacency, options=options)

    monkeypatch.setattr(pymetis, "part_graph", spy)
    neighbors = [np.array([1]), np.array([0, 2]), np.array([1, 3]), np.array([2])]
    state = np.random.get_state()
    a, count = metis_partition_cells(neighbors, 2, seed=seed)
    b, _ = metis_partition_cells(neighbors, 2, seed=seed)
    assert calls == [seed, seed]
    np.testing.assert_array_equal(a, b)
    assert count == 2
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])
    if seed == -1:
        assert pymetis.Options().seed == -1
        np.testing.assert_array_equal(
            a, real(2, adjacency=[n.tolist() for n in neighbors])[1]
        )
