from pathlib import Path

import pytest

from dtcc_sim import fenics


class _FakeTopology:
    dim = 3

    def __init__(self):
        self.entities_calls = []
        self.connectivity_calls = []

    def create_entities(self, dim):
        self.entities_calls.append(dim)

    def create_connectivity(self, d0, d1):
        self.connectivity_calls.append((d0, d1))


class _FakeMesh:
    def __init__(self):
        self.topology = _FakeTopology()


def test_load_mesh_accepts_pathlike(monkeypatch):
    opened = []
    fake_mesh = _FakeMesh()

    class _FakeXDMFFile:
        def __init__(self, comm, filename, mode):
            opened.append((filename, mode))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read_mesh(self, name):
            assert name == "mesh"
            return fake_mesh

    monkeypatch.setattr(fenics, "XDMFFile", _FakeXDMFFile)

    mesh = fenics.load_mesh(Path("/tmp/test_mesh.xdmf"))

    assert mesh is fake_mesh
    assert opened == [("/tmp/test_mesh.xdmf", "r")]


def test_load_mesh_with_markers_accepts_pathlike(monkeypatch):
    opened = []
    fake_mesh = _FakeMesh()
    fake_markers = object()

    class _FakeXDMFFile:
        def __init__(self, comm, filename, mode):
            opened.append((filename, mode))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read_mesh(self, name):
            assert name == "mesh"
            return fake_mesh

        def read_meshtags(self, mesh, name):
            assert mesh is fake_mesh
            assert name == "boundary_markers"
            return fake_markers

    monkeypatch.setattr(fenics, "XDMFFile", _FakeXDMFFile)

    mesh, markers = fenics.load_mesh_with_markers(Path("/tmp/test_mesh.xdmf"))

    assert mesh is fake_mesh
    assert markers is fake_markers
    assert opened == [("/tmp/test_mesh.xdmf", "r")]
    assert fake_mesh.topology.entities_calls == [2]
    assert fake_mesh.topology.connectivity_calls == [(2, 3)]


def test_load_mesh_rejects_non_xdmf_pathlike():
    with pytest.raises(ValueError, match="filename must end with \\.xdmf"):
        fenics.load_mesh(Path("/tmp/test_mesh.pb"))
