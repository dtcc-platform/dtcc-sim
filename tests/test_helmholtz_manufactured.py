"""Manufactured plane waves check the PDE, boundary data and P1 accuracy."""

import numpy as np
import pytest

pytest.importorskip("dolfinx")
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh as dmesh
import ufl

from dtcc_sim.helmholtz._backend.params import Params, bind_parameters
from dtcc_sim.helmholtz._backend import mesh as backend


def _plane_wave_error(n, boundary, coefficients, monkeypatch):
    mesh = dmesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
    tags = None
    if boundary != "all_impedance":
        fdim = mesh.topology.dim - 1
        facets = np.sort(
            dmesh.locate_entities_boundary(
                mesh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
            )
        )
        midpoints = dmesh.compute_midpoints(mesh, fdim, facets)
        labels = np.full(len(facets), 25, dtype=np.int32)
        labels[np.isclose(midpoints[:, 0], 0)] = -2
        labels[np.isclose(midpoints[:, 0], 1)] = 17
        tags = dmesh.meshtags(mesh, fdim, facets, labels)
    monkeypatch.setattr(backend, "load_mesh", lambda: (mesh, tags, None))
    A_const, V_const, beta_const = coefficients
    params = Params(
        rhs_mode="plane_wave_exact",
        boundary_mode="mixed" if boundary == "mixed" else "impedance_only",
        impedance_facet_tags=(-2,),
        dirichlet_facet_tags=(17,) if boundary == "mixed" else (),
        frequency_hz=343 * 1.5 / (2 * np.pi),
        sound_speed=343,
        A_const=A_const,
        V_const=V_const,
        beta_const=beta_const,
        plane_wave_direction_x=1,
        plane_wave_direction_y=0.3,
        plane_wave_direction_z=0.2,
        verbose=False,
    )
    with bind_parameters(params):
        system = backend.build_system()
        V = system[3]
        A, b = system[-2:]
        assert b.norm() > 1e-3, (
            "Manufactured boundary data must produce a nonzero right hand side"
        )
        solution = A.createVecRight()
        solver = PETSc.KSP().create(mesh.comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.solve(b, solution)
        assert solver.getConvergedReason() > 0
        residual = b.duplicate()
        A.mult(solution, residual)
        residual.axpy(-1, b)
        assert residual.norm() / b.norm() < 1e-10
        uh = fem.Function(V)
        uh.x.array[:] = solution.getArray(readonly=True)
        uh.x.scatter_forward()
        exact = backend._plane_wave_expression(mesh)
        dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 8})
        local_error = fem.assemble_scalar(
            fem.form(ufl.inner(uh - exact, uh - exact) * dx)
        ).real
        local_norm = fem.assemble_scalar(fem.form(ufl.inner(exact, exact) * dx)).real
        error = float(
            np.sqrt(mesh.comm.allreduce(local_error) / mesh.comm.allreduce(local_norm))
        )
        solver.destroy()
        solution.destroy()
        residual.destroy()
        A.destroy()
        b.destroy()
    return error


@pytest.mark.parametrize("boundary", ["all_impedance", "tagged_impedance", "mixed"])
@pytest.mark.parametrize("coefficients", [(1.0, 1.0, 1.0), (1.7, 0.8, 0.9)])
def test_plane_wave_accuracy_improves_under_refinement(
    monkeypatch, boundary, coefficients, record_property
):
    coarse = _plane_wave_error(4, boundary, coefficients, monkeypatch)
    fine = _plane_wave_error(8, boundary, coefficients, monkeypatch)
    record_property("coarse_relative_L2_error", coarse)
    record_property("fine_relative_L2_error", fine)
    assert fine < 0.02
    assert fine < 0.4 * coarse, (coarse, fine)
