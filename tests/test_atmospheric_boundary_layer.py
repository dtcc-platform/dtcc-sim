"""Atmospheric boundary-layer closures, wall law and equilibrium inflow."""

import numpy as np
import pytest
from pydantic import ValidationError

from dtcc_sim import NavierStokesParameters, StokesParameters

pytestmark = [pytest.mark.simulation]


def abl(**kwargs):
    return dict(inlet_profile="log_law", side_top_boundary="slip", **kwargs)


class TestParameters:
    def test_default_keeps_constant_viscosity(self):
        params = NavierStokesParameters()
        assert params.turbulence_model == "constant"
        assert not params.uses_wall_distance
        assert params.wall_offset == params.z0

    def test_friction_velocity_reproduces_reference_speed(self):
        for wall_model in ["noslip", "log_law"]:
            params = NavierStokesParameters(**abl(turbulence_model="abl", wall_model=wall_model,
                                                  wind_speed=5., z0=.5, u_ref_height=10.))
            speed = params.friction_velocity / .41 * np.log((10. + params.wall_offset) / .5)
            assert speed == pytest.approx(5.)
        assert params.wall_offset == pytest.approx(.5 + params.wall_function_height)

    @pytest.mark.parametrize("kwargs, message", [
        (dict(turbulence_model="abl", inlet_profile="power_law"), "requires inlet_profile='log_law'"),
        (dict(turbulence_model="mixing_length", inlet_profile="log_law", nu_t=1.), "nu_t applies only"),
        (dict(wall_model="log_law", inlet_profile="uniform"), "wall_model='log_law' requires"),
    ])
    def test_inconsistent_closures_are_rejected(self, kwargs, message):
        with pytest.raises(ValidationError, match=message):
            NavierStokesParameters(**kwargs)

    def test_stokes_rejects_mixing_length(self):
        with pytest.raises(ValidationError, match="Navier–Stokes Newton"):
            StokesParameters(**abl(turbulence_model="mixing_length"))
        assert StokesParameters(**abl(turbulence_model="abl", wall_model="log_law")).uses_wall_distance

    def test_log_law_inlet_uses_wall_offset(self):
        from dtcc_sim._flow.solver import make_inlet_velocity_expression
        params = NavierStokesParameters(inlet_profile="log_law", wall_model="log_law", wind_speed=5.)
        values = make_inlet_velocity_expression(params)(np.array([[0., 0.], [0., 0.], [0., 10.]]))
        zw = params.wall_offset
        assert values[0, 1] == pytest.approx(5.)
        assert values[0, 0] == pytest.approx(5. * np.log(zw / params.z0) / np.log((10. + zw) / params.z0))


# ---------------------------------------------------------------------------
# Numerical checks
# ---------------------------------------------------------------------------

def _numerics():
    pytest.importorskip("dolfinx")
    from petsc4py import PETSc
    if not np.issubdtype(PETSc.ScalarType, np.floating):
        pytest.skip("Stationary flow numerical tests require real PETSc")


def tagged_box(extent, cells, slope=0.):
    """Box with canonical DTCC markers: ground -1, top -2, x faces -3/-4, y faces -5/-6."""
    import dolfinx.mesh as dm
    from mpi4py import MPI
    mesh = dm.create_box(MPI.COMM_WORLD, [[0., 0., 0.], list(extent)], list(cells))
    mesh.topology.create_entities(2)
    mesh.topology.create_connectivity(2, 3)
    faces = dm.exterior_facet_indices(mesh.topology)
    mid = dm.compute_midpoints(mesh, 2, faces)
    values = np.zeros(len(faces), dtype=np.int32)
    for axis, coordinate, marker in [(2, 0., -1), (2, extent[2], -2), (0, 0., -3),
                                     (0, extent[0], -4), (1, 0., -5), (1, extent[1], -6)]:
        values[np.isclose(mid[:, axis], coordinate)] = marker
    if slope:
        x = mesh.geometry.x
        x[:, 2] += slope * (x[:, 0] + .5 * x[:, 1]) * (1 - x[:, 2] / extent[2])
    order = np.argsort(faces)
    return mesh, dm.meshtags(mesh, 2, faces[order], values[order])


def integral(mesh, expression):
    import dolfinx.fem as fem
    from mpi4py import MPI
    return float(mesh.comm.allreduce(fem.assemble_scalar(fem.form(expression)), op=MPI.SUM))


@pytest.mark.fenics
@pytest.mark.slow
def test_wall_distance_is_exact_for_flat_ground():
    _numerics()
    import ufl
    from dtcc_sim._flow.solver import BndCat, _wall_distance, categorize_boundary
    mesh, tags = tagged_box((4., 2., 3.), (4, 2, 3))
    categories = categorize_boundary(mesh, tags, 0, -3, -4)
    _, distance = _wall_distance(mesh, categories, [BndCat.GROUND], "ground")
    z = ufl.SpatialCoordinate(mesh)[2]
    assert np.sqrt(integral(mesh, (distance - z)**2 * ufl.dx)) < 1e-10


@pytest.mark.fenics
@pytest.mark.slow
@pytest.mark.parametrize("turbulence_model", ["abl", "mixing_length"])
def test_jacobian_with_closure_and_weak_terrain_walls(turbulence_model):
    """Newton derivative of the closure, wall law and Nitsche terms on sloped ground."""
    _numerics()
    import dolfinx.fem as fem
    from dolfinx.fem import petsc as fp
    from dtcc_sim._flow.solver import _MixedFlowSystem, categorize_boundary
    mesh, tags = tagged_box((40., 20., 20.), (4, 2, 2), slope=.15)
    params = NavierStokesParameters(**abl(turbulence_model=turbulence_model, wall_model="log_law",
                                          backflow_beta=2., grad_div_gamma=3.))
    system = _MixedFlowSystem(mesh, params, tags, categorize_boundary(mesh, tags, 0, -3, -4), None)
    system.set_stage(1.)
    rng = np.random.default_rng(7)
    system.w.x.array[:] = 3 * rng.normal(size=system.w.x.array.size)
    matrix = fp.assemble_matrix(fem.form(system.J))
    matrix.assemble()
    direction = system.w.x.petsc_vec.duplicate()
    direction.array[:] = rng.normal(size=direction.getLocalSize())
    direction.normalize()
    action = direction.duplicate()
    matrix.mult(direction, action)
    residual, original, values = fem.form(system.F), system.w.x.array.copy(), []
    for sign in [1, -1]:
        system.w.x.array[:] = original
        system.w.x.petsc_vec.axpy(sign * 1e-6, direction)
        system.w.x.scatter_forward()
        values.append(fp.assemble_vector(residual))
    values[0].axpy(-1, values[1])
    values[0].scale(1 / 2e-6)
    values[0].axpy(-1, action)
    assert values[0].norm() / action.norm() < 1e-6


def weak_manufactured(n, turbulence_model):
    """Exact solution enforced through the weak residual of the full formulation."""
    import ufl
    from dtcc_sim._flow.solver import DEFAULT_STATIONARY_PETSC, _MixedFlowSystem, categorize_boundary
    mesh, tags = tagged_box((2., 1., 1.), (2 * n, n, n))
    params = NavierStokesParameters(**abl(
        turbulence_model=turbulence_model, wall_model="log_law", wind_speed=1., z0=.05,
        u_ref_height=.5, wall_function_height=.1, nu=1e-3,
    ))
    x = ufl.SpatialCoordinate(mesh)
    bubble = x[2]**2 * (1 - x[2])**2
    exact_u = ufl.as_vector((1 + .5 * bubble.dx(2) * ufl.sin(ufl.pi * x[0]), 0.,
                             -.5 * bubble * ufl.pi * ufl.cos(ufl.pi * x[0])))
    exact_p = ufl.sin(x[0]) * ufl.cos(x[2])

    def inlet(points):
        z = points[2]
        return np.array([1 + .5 * (2 * z * (1 - z)**2 - 2 * z**2 * (1 - z)) * np.sin(np.pi * points[0]),
                         np.zeros_like(z), -.5 * z**2 * (1 - z)**2 * np.pi * np.cos(np.pi * points[0])])

    system = _MixedFlowSystem(mesh, params, tags, categorize_boundary(mesh, tags, 0, -3, -4), inlet)
    exact = ufl.as_vector([exact_u[0], exact_u[1], exact_u[2], exact_p])
    system.F -= ufl.replace(system.F, {system.w: exact})
    system.J = ufl.derivative(system.F, system.w)
    _, dp = ufl.TrialFunctions(system.W)
    _, q = ufl.TestFunctions(system.W)
    system.P = system.J + dp * q * ufl.dx
    system.configure({**DEFAULT_STATIONARY_PETSC, "snes_type": "newtonls", "snes_linesearch_type": "bt",
                      "snes_rtol": 1e-11, "snes_atol": 1e-12, "snes_max_it": 30})
    return system, exact_u, exact_p


@pytest.mark.fenics
@pytest.mark.slow
@pytest.mark.parametrize("turbulence_model", ["abl", "mixing_length"])
def test_variable_viscosity_manufactured_convergence(turbulence_model):
    """Symmetric stress, open-boundary correction, wall law and top shear converge at Taylor–Hood rates."""
    _numerics()
    import ufl
    errors = []
    for n in [2, 4, 8]:
        system, exact_u, exact_p = weak_manufactured(n, turbulence_model)
        try:
            system.solve(0.)
            system.solve(1.)
            u, p = system.fields()
            errors.append([np.sqrt(system.integral(ufl.inner(u - exact_u, u - exact_u) * ufl.dx)),
                           np.sqrt(system.integral((p - exact_p)**2 * ufl.dx))])
        finally:
            system.close()
    errors = np.array(errors)
    orders = np.log2(errors[1] / errors[2])
    assert np.all(orders > [2.5, 1.5]), (errors, orders)


@pytest.mark.fenics
@pytest.mark.slow
@pytest.mark.parametrize("solver_name", ["navierstokes", "stokes"])
def test_equilibrium_inflow_is_horizontally_homogeneous(solver_name):
    """An empty flat domain preserves the equilibrium log-law inflow."""
    _numerics()
    from dolfinx import geometry
    from dtcc_sim import NavierStokesSolver, StokesSolver
    from dtcc_sim._flow.parameters import KAPPA
    mesh, tags = tagged_box((200., 20., 80.), (20, 2, 8))
    parameter_type, solver_type = ((NavierStokesParameters, NavierStokesSolver) if solver_name == "navierstokes"
                                   else (StokesParameters, StokesSolver))
    params = parameter_type(**abl(turbulence_model="abl", wall_model="log_law", wind_speed=5.))
    solver = solver_type(mesh=mesh, markers=tags, params=params)
    u, _ = solver.simulate()
    heights = np.array([5., 10., 20., 40., 75.])
    exact = params.friction_velocity / KAPPA * np.log((heights + params.wall_offset) / params.z0)
    points = np.column_stack([np.full_like(heights, 200. - 1e-6), np.full_like(heights, 10.), heights])
    tree = geometry.bb_tree(mesh, 3)
    cells = geometry.compute_colliding_cells(mesh, geometry.compute_collisions_points(tree, points), points)
    outlet = u.eval(points, np.array([cells.links(i)[0] for i in range(len(points))], dtype=np.int32))
    assert np.max(np.abs(outlet[:, 0] - exact) / exact) < .01
    assert np.max(np.abs(outlet[:, 1:])) < .01 * params.wind_speed
    turbulence = solver.diagnostics["turbulence"]
    assert turbulence["viscous_form"] == "symmetric"
    assert turbulence["top_shear_stress"] == pytest.approx(params.friction_velocity**2)
    assert turbulence["wall_distance_m"]["mean"] == pytest.approx(40., rel=1e-8)


@pytest.mark.fenics
@pytest.mark.slow
def test_inflow_is_tangential_on_sloping_weak_walls():
    """Inlet edges on sloping ground admit no wall-normal inflow; interior inflow is unchanged."""
    _numerics()
    import dolfinx
    from dtcc_sim._flow.solver import BndCat, _tangential_inflow, categorize_boundary
    mesh, tags = tagged_box((40., 20., 20.), (4, 2, 2), slope=.3)
    categories = categorize_boundary(mesh, tags, 0, -3, -4)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (3,)))
    inlet = dolfinx.fem.Function(V)
    inlet.interpolate(lambda x: np.vstack([np.full(x.shape[1], 5.), np.zeros(x.shape[1]), np.zeros(x.shape[1])]))
    _tangential_inflow(mesh, categories, [BndCat.WALL, BndCat.ROOF, BndCat.GROUND], inlet)
    values = inlet.x.array.reshape(-1, 3)
    fd = mesh.topology.dim - 1
    inlet_dofs = dolfinx.fem.locate_dofs_topological(V, fd, categories.find(int(BndCat.INLET)))
    ground = categories.find(int(BndCat.GROUND))
    edge = np.intersect1d(inlet_dofs, dolfinx.fem.locate_dofs_topological(V, fd, ground))
    assert len(edge)
    points = mesh.geometry.x[dolfinx.mesh.entities_to_geometry(mesh, fd, ground)]
    normal = np.cross(points[0, 1] - points[0, 0], points[0, 2] - points[0, 0])
    normal /= np.linalg.norm(normal)
    assert np.max(np.abs(values[edge] @ normal)) < 1e-12
    assert np.min(values[edge, 0]) > 4.
    interior = np.setdiff1d(inlet_dofs, edge)
    np.testing.assert_array_equal(values[interior], [[5., 0., 0.]] * len(interior))
