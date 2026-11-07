"""Atmospheric boundary-layer closures, wall law and equilibrium inflow."""

import numpy as np
import pytest
from pydantic import ValidationError

from dtcc_sim.navierstokes.parameters import NavierStokesParameters
from dtcc_sim.stokes.parameters import StokesParameters

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
        pytest.importorskip("dolfinx")
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


def tagged_box(extent, cells, slope=0., comm=None):
    """Box with canonical DTCC markers: ground -1, top -2, x faces -3/-4, y faces -5/-6."""
    import dolfinx.mesh as dm
    from mpi4py import MPI
    mesh = dm.create_box(comm or MPI.COMM_WORLD, [[0., 0., 0.], list(extent)], list(cells))
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
    from petsc4py import PETSc
    from dtcc_sim._flow.solver import _MixedFlowSystem, categorize_boundary
    mesh, tags = tagged_box((40., 20., 20.), (4, 2, 2), slope=.15)
    params = NavierStokesParameters(**abl(turbulence_model=turbulence_model, wall_model="log_law",
                                          backflow_beta=2., grad_div_gamma=3.))
    system = _MixedFlowSystem(mesh, params, tags, categorize_boundary(mesh, tags, 0, -3, -4), None)
    system.set_stage(1.)
    rng = np.random.default_rng(7 + mesh.comm.rank)
    system.w.x.array[:] = 3 * rng.normal(size=system.w.x.array.size)
    system.w.x.scatter_forward()
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
        vector = fp.assemble_vector(residual)
        vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        values.append(vector)
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


def independent_manufactured(n, turbulence_model):
    """Manufactured solution with independently derived forcing and boundary data.

    The viscosity, wall drag and expected boundary terms are written out from
    the model equations; the implementation's terms must reproduce them for
    the discrete solution to converge to the exact one.
    """
    import ufl
    from dtcc_sim._flow.parameters import KAPPA
    from dtcc_sim._flow.solver import BndCat, DEFAULT_STATIONARY_PETSC, _MixedFlowSystem, categorize_boundary
    mesh, tags = tagged_box((2., 1., 1.), (2 * n, n, n))
    nu, z0, delta, z_ref, speed = 1e-3, .05, .1, .5, 1.
    params = NavierStokesParameters(**abl(
        turbulence_model=turbulence_model, wall_model="log_law", wind_speed=speed, z0=z0,
        u_ref_height=z_ref, wall_function_height=delta, nu=nu,
    ))
    x = ufl.SpatialCoordinate(mesh)
    bubble = x[2]**2 * (1 - x[2])**2
    exact_u = ufl.as_vector((1 + x[2] + .1 * bubble.dx(2) * ufl.sin(ufl.pi * x[0]), 0.,
                             -.1 * bubble * ufl.pi * ufl.cos(ufl.pi * x[0])))
    exact_p = ufl.sin(x[0]) * ufl.cos(x[2])

    def inlet(points):
        z = points[2]
        return np.array([1 + z + .1 * (2 * z * (1 - z)**2 - 2 * z**2 * (1 - z)) * np.sin(np.pi * points[0]),
                         np.zeros_like(z), -.1 * z**2 * (1 - z)**2 * np.pi * np.cos(np.pi * points[0])])

    # Model data written out independently: flat ground at z = 0 is the only wall.
    z_w = z0 + delta
    friction = KAPPA * speed / np.log((z_ref + z_w) / z0)
    length = KAPPA * (x[2] + z_w)
    strain = ufl.sym(ufl.grad(exact_u))
    eddy = friction * length if turbulence_model == "abl" else length**2 * ufl.sqrt(2 * ufl.inner(strain, strain))
    stress = 2 * (nu + eddy) * strain
    forcing = -ufl.div(stress) + ufl.grad(exact_u) * exact_u + ufl.grad(exact_p)
    drag = (KAPPA / np.log(z_w / z0))**2 * ufl.sqrt(exact_u[0]**2 + exact_u[1]**2)
    top_shear = friction**2  # along x for wind from 270 degrees

    system = _MixedFlowSystem(mesh, params, tags, categorize_boundary(mesh, tags, 0, -3, -4), inlet)
    v, _ = ufl.TestFunctions(system.W)
    n_ = ufl.FacetNormal(mesh)
    dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 8})
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=system.ds.subdomain_data(),
                     metadata={"quadrature_degree": 8})
    traction = stress * n_
    outlet = (nu + eddy) * ufl.grad(exact_u) * n_ - exact_p * n_  # gradient-form outflow traction
    top = ufl.as_vector((traction[0] - top_shear, traction[1], 0.))
    ground = ufl.as_vector((traction[0] + drag * exact_u[0], traction[1] + drag * exact_u[1], 0.))
    system.F -= (ufl.inner(forcing, v) * dx + ufl.inner(outlet, v) * ds(int(BndCat.OUTLET))
                 + ufl.inner(top, v) * ds(int(BndCat.TOP)) + ufl.inner(ground, v) * ds(int(BndCat.GROUND)))
    system.J = ufl.derivative(system.F, system.w)
    _, dp = ufl.TrialFunctions(system.W)
    _, q = ufl.TestFunctions(system.W)
    system.P = system.J + dp * q * ufl.dx
    system.configure({**DEFAULT_STATIONARY_PETSC, "snes_type": "newtonls", "snes_linesearch_type": "bt",
                      "snes_rtol": 1e-11, "snes_atol": 1e-12, "snes_max_it": 30})
    return system, exact_u, exact_p


@pytest.mark.fenics
@pytest.mark.slow
@pytest.mark.parametrize("construction", ["weak", "independent"])
@pytest.mark.parametrize("turbulence_model", ["abl", "mixing_length"])
def test_variable_viscosity_manufactured_convergence(turbulence_model, construction):
    """Symmetric stress, open-boundary correction, wall law and top shear converge at Taylor–Hood rates.

    The weak construction checks the discretization of the implemented
    residual; the independent construction checks that residual against
    forcing and boundary data derived from the model equations.
    """
    _numerics()
    import ufl
    manufactured = weak_manufactured if construction == "weak" else independent_manufactured
    errors = []
    for n in [2, 4, 8]:
        system, exact_u, exact_p = manufactured(n, turbulence_model)
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
@pytest.mark.parametrize("solver_name, turbulence_model", [
    ("navierstokes", "abl"), ("stokes", "abl"), ("navierstokes", "mixing_length")])
def test_equilibrium_inflow_is_horizontally_homogeneous(solver_name, turbulence_model):
    """An empty flat domain preserves the equilibrium log-law inflow."""
    _numerics()
    from dolfinx import geometry
    from dtcc_sim import NavierStokesSolver, StokesSolver
    from dtcc_sim._flow.parameters import KAPPA
    mesh, tags = tagged_box((200., 20., 80.), (20, 2, 8))
    parameter_type, solver_type = ((NavierStokesParameters, NavierStokesSolver) if solver_name == "navierstokes"
                                   else (StokesParameters, StokesSolver))
    params = parameter_type(**abl(turbulence_model=turbulence_model, wall_model="log_law", wind_speed=5.))
    solver = solver_type(mesh=mesh, markers=tags, params=params)
    u, _ = solver.simulate()
    heights = np.array([5., 10., 20., 40., 75.])
    exact = params.friction_velocity / KAPPA * np.log((heights + params.wall_offset) / params.z0)
    points = np.column_stack([np.full_like(heights, 200. - 1e-6), np.full_like(heights, 10.), heights])
    tree = geometry.bb_tree(mesh, 3)
    cells = geometry.compute_colliding_cells(mesh, geometry.compute_collisions_points(tree, points), points)
    found = [i for i in range(len(points)) if len(cells.links(i))]
    local = np.full((len(points), 3), np.nan)
    if found:
        local[found] = u.eval(points[found], np.array([cells.links(i)[0] for i in found], dtype=np.int32))
    outlet = np.nanmean(np.stack(mesh.comm.allgather(local)), axis=0)  # from the ranks holding each point
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
    assert mesh.comm.allreduce(len(edge)) > 0
    normal = np.array([-.3, -.15, 1.]) / np.linalg.norm([-.3, -.15, 1.])  # ground z = 0.3 (x + y/2)
    assert np.all(np.abs(values[edge] @ normal) < 1e-12)
    assert np.all(values[edge, 0] > 4.)
    interior = np.setdiff1d(inlet_dofs, edge)
    np.testing.assert_array_equal(values[interior], np.tile([5., 0., 0.], (len(interior), 1)))


def test_lateral_faces_follow_the_wind_direction():
    pytest.importorskip("dolfinx")
    from dtcc_sim._flow.solver import classify_lateral_faces
    cases = {270.: ([-3], [-4]), 180.: ([-5], [-6]), 250.: ([-3, -5], [-4, -6]), 45.: ([-4, -6], [-3, -5])}
    for direction, (inlets, outlets) in cases.items():
        found = classify_lateral_faces(NavierStokesParameters(wind_dir_deg=direction))
        assert (sorted(found[0]), sorted(found[1])) == (sorted(inlets), sorted(outlets)), direction
    explicit = NavierStokesParameters(wind_dir_deg=250., inlet_marker=-3, outlet_marker=-4)
    assert classify_lateral_faces(explicit) == ([-3], [-4])
    assert classify_lateral_faces(NavierStokesParameters(closed_cavity=True)) == ([-3], [])


@pytest.mark.fenics
@pytest.mark.slow
@pytest.mark.parametrize("turbulence_model", ["abl", "mixing_length"])
def test_oblique_wind_keeps_its_direction(turbulence_model):
    """Wind 20 degrees off the x-axis enters and leaves through two faces each."""
    _numerics()
    import ufl
    from dtcc_sim import NavierStokesSolver
    from dtcc_sim._flow.parameters import KAPPA
    mesh, tags = tagged_box((160., 160., 80.), (8, 8, 4))
    params = NavierStokesParameters(**abl(turbulence_model=turbulence_model, wall_model="log_law",
                                          wind_speed=5., wind_dir_deg=250.))
    solver = NavierStokesSolver(mesh=mesh, markers=tags, params=params)
    u, _ = solver.simulate()
    mean = [integral(mesh, u[i] * ufl.dx) for i in range(2)]
    assert np.degrees(np.arctan2(mean[1], mean[0])) == pytest.approx(20., abs=.1)
    z = ufl.SpatialCoordinate(mesh)[2]
    speed = params.friction_velocity / KAPPA * ufl.ln((z + params.wall_offset) / params.z0)
    wx, wy = params.wind_vector_xy
    exact = ufl.as_vector((speed * wx, speed * wy, 0.))
    deviation = integral(mesh, ufl.inner(u - exact, u - exact) * ufl.dx) / integral(mesh, ufl.inner(exact, exact) * ufl.dx)
    assert np.sqrt(deviation) < .02
    assert sorted(solver.diagnostics["inlet_markers"]) == [-5, -3]
    assert sorted(solver.diagnostics["outlet_markers"]) == [-6, -4]


@pytest.mark.fenics
@pytest.mark.slow
def test_weak_wall_leakage_is_measured_and_gated():
    """Flow through weakly enforced curved terrain is bounded by the leakage gates."""
    _numerics()
    from dtcc_sim import NavierStokesSolver

    def hills():
        mesh, tags = tagged_box((200., 40., 80.), (20, 4, 8))
        x = mesh.geometry.x
        x[:, 2] += 4 * np.sin(2 * np.pi * x[:, 0] / 80) * np.sin(2 * np.pi * x[:, 1] / 80) * (1 - x[:, 2] / 80)
        return mesh, tags

    params = NavierStokesParameters(**abl(turbulence_model="abl", wall_model="log_law", wind_speed=5.,
                                          divergence_tolerance=1., flux_imbalance_tolerance=1e-3))
    mesh, tags = hills()
    solver = NavierStokesSolver(mesh=mesh, markers=tags, params=params)
    solver.simulate()
    convergence = solver.diagnostics["convergence"]
    assert 0 < convergence["ground_leakage_fraction"] < params.ground_leakage_tolerance
    assert convergence["building_leakage_fraction"] == 0.
    assert convergence["ground_normal_velocity_rms"] > 0
    strict = params.model_copy(update={"ground_leakage_tolerance": 1e-6})
    mesh, tags = hills()
    solver = NavierStokesSolver(mesh=mesh, markers=tags, params=strict)
    with pytest.raises(RuntimeError, match="quality checks failed"):
        solver.simulate()
    assert solver.diagnostics["stop_reason"] == "quality_failure"


@pytest.mark.fenics
@pytest.mark.slow
def test_tangential_inflow_is_partition_independent():
    """Inflow projected on curved terrain matches a serial projection on every rank."""
    _numerics()
    import dolfinx
    from mpi4py import MPI
    from dtcc_sim._flow.solver import BndCat, _tangential_inflow, categorize_boundary

    def projected(comm):
        mesh, tags = tagged_box((200., 40., 80.), (20, 4, 8), comm=comm)
        x = mesh.geometry.x
        # Ground curved along the inlet and rising into the domain
        x[:, 2] += 3 * np.sin(2 * np.pi * x[:, 0] / 50) * (1 + x[:, 1] / 20) * (1 - x[:, 2] / 80)
        categories = categorize_boundary(mesh, tags, 0, -5, -6)  # wind from the south
        V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (3,)))
        inlet = dolfinx.fem.Function(V)
        inlet.interpolate(lambda p: np.vstack([np.zeros(p.shape[1]), 5 + .01 * p[2], np.zeros(p.shape[1])]))
        _tangential_inflow(mesh, categories, [BndCat.WALL, BndCat.ROOF, BndCat.GROUND], inlet)
        dofs = dolfinx.fem.locate_dofs_topological(V, 2, categories.find(int(BndCat.INLET)))
        keys = [tuple(row) for row in np.round(V.tabulate_dof_coordinates()[dofs], 8)]
        return dict(zip(keys, inlet.x.array.reshape(-1, 3)[dofs]))

    distributed, serial = projected(MPI.COMM_WORLD), projected(MPI.COMM_SELF)
    assert set(distributed) <= set(serial)
    changed = sum(np.linalg.norm(serial[key] - [0., 5 + .01 * key[2], 0.]) > 1e-6 for key in serial)
    assert changed > 20
    assert max(np.linalg.norm(value - serial[key]) for key, value in distributed.items()) < 1e-12


@pytest.mark.fenics
@pytest.mark.slow
def test_wall_normals_are_averaged_and_partition_independent():
    """Averaged wall normals follow curved terrain, keep sharp edges and match a serial computation."""
    _numerics()
    from mpi4py import MPI
    from dtcc_sim._flow.solver import BndCat, _wall_normals, categorize_boundary

    def normals(comm):
        mesh, tags = tagged_box((200., 40., 80.), (20, 4, 8), comm=comm)
        x = mesh.geometry.x
        x[:, 2] += 1.5 * np.sin(2 * np.pi * x[:, 0] / 100) * (1 + x[:, 1] / 40) * (1 - x[:, 2] / 80)
        categories = categorize_boundary(mesh, tags, 0, -5, -6)
        field, _ = _wall_normals(mesh, categories, [BndCat.WALL, BndCat.ROOF, BndCat.GROUND])
        coordinates = field.function_space.tabulate_dof_coordinates()
        values = field.x.array.reshape(-1, 3)
        wall = np.linalg.norm(values, axis=1) > 0
        return dict(zip([tuple(row) for row in np.round(coordinates[wall], 8)], values[wall]))

    distributed, serial = normals(MPI.COMM_WORLD), normals(MPI.COMM_SELF)
    assert set(distributed) <= set(serial) and len(serial) == 21 * 5  # every ground vertex
    assert max(np.linalg.norm(value - serial[key]) for key, value in distributed.items()) < 1e-12
    # Outward ground normals point down; the analytic normal of z = g(x, y) is (g_x, g_y, -1) normalized.
    for (x, y, z), value in serial.items():
        gx = 1.5 * 2 * np.pi / 100 * np.cos(2 * np.pi * x / 100) * (1 + y / 40)
        gy = 1.5 * np.sin(2 * np.pi * x / 100) / 40
        exact = np.array([gx, gy, -1.]) / np.linalg.norm([gx, gy, -1.])
        assert np.dot(value, exact) > np.cos(np.deg2rad(2.))  # one-sided at the domain edges
