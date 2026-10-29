"""Stationary mixed-flow accuracy, pressure gauges and failure contracts."""

import numpy as np
import pytest

pytest.importorskip("dolfinx")
import dolfinx.fem as fem
from dolfinx.fem import petsc as fp
import dolfinx.mesh as dm
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import basix.ufl

from dtcc_sim._flow.solver import (
    BndCat, _MixedFlowSystem, categorize_boundary,
)

from dtcc_sim import NavierStokesParameters, NavierStokesSolver, StokesParameters, StokesSolver

pytestmark = [pytest.mark.fenics, pytest.mark.simulation, pytest.mark.slow]


@pytest.fixture(autouse=True)
def real_runtime():
    if not np.issubdtype(PETSc.ScalarType, np.floating):
        pytest.skip("Stationary flow numerical tests require real PETSc")


def box_case(n=3, closed=False):
    mesh = dm.create_box(MPI.COMM_WORLD, [[0., 0., 0.], [2., 1., 1.]], [2*n, n, n])
    mesh.topology.create_entities(2)
    mesh.topology.create_connectivity(2, 3)
    faces = dm.exterior_facet_indices(mesh.topology)
    mid = dm.compute_midpoints(mesh, 2, faces)
    values = np.full(len(faces), -1, dtype=np.int32)
    for axis, coordinate, marker in [(0, 0, -3)] + ([] if closed else [(0, 2, -4), (1, 0, -5), (1, 1, -6)]):
        values[np.isclose(mid[:, axis], coordinate)] = marker
    order = np.argsort(faces)
    return mesh, dm.meshtags(mesh, 2, faces[order], values[order])


def parabolic(x):
    return np.array([4*x[2]*(1-x[2]), np.zeros(x.shape[1]), np.zeros(x.shape[1])])


def moving_wall(x):
    return np.array([np.zeros(x.shape[1]), .1*x[1]*(1-x[1])*x[2]*(1-x[2]), np.zeros(x.shape[1])])


def integral(mesh, expression):
    return float(mesh.comm.allreduce(fem.assemble_scalar(fem.form(expression)), op=MPI.SUM))


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("solver_type,parameter_type", [(NavierStokesSolver, NavierStokesParameters), (StokesSolver, StokesParameters)])
def test_exact_planar_poiseuille_and_traction(solver_type, parameter_type, direct):
    mesh, tags = box_case(4)
    options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"} if direct else None
    key = "petsc_options" if parameter_type is NavierStokesParameters else "petsc_stokes"
    params = parameter_type(nu=.1, wind_speed=1., side_top_boundary="slip", **{key: options})
    solver = solver_type(mesh=mesh, markers=tags, params=params, inlet_expression=parabolic)
    u, p = solver.simulate()
    x = ufl.SpatialCoordinate(mesh)
    exact = ufl.as_vector((4*x[2]*(1-x[2]), 0., 0.))
    assert np.sqrt(integral(mesh, ufl.inner(u-exact, u-exact)*ufl.dx)) < 1e-7
    assert np.sqrt(integral(mesh, (p-.8*(2-x[0]))**2*ufl.dx)) < 1e-7
    diagnostics = solver.diagnostics
    assert diagnostics["pressure_gauge"] == "traction_determined"
    assert diagnostics["stop_reason"] == "converged"
    assert diagnostics["convergence"]["dimensionless_divergence"] < .005
    assert diagnostics["convergence"]["flux_imbalance"] < 1e-4
    assert diagnostics["convergence"]["residual_norm"] <= diagnostics["convergence"]["residual_limit"]
    assert diagnostics["convergence"]["wall_normal_velocity_rms"] < 1e-10
    assert diagnostics["convergence"]["slip_normal_velocity_rms"] < 1e-10


@pytest.mark.parametrize("direct", [False, True])
def test_closed_pressure_nullspace_and_gauge(direct):
    mesh, tags = box_case(3, closed=True)
    tags = dm.meshtags(mesh, 2, tags.indices, np.full(len(tags.indices), -3, dtype=np.int32))
    options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_24": 1, "mat_mumps_icntl_25": 0} if direct else None
    params = NavierStokesParameters(nu=.1, closed_cavity=True, side_top_boundary="slip", petsc_options=options)
    def rotation(x):
        return np.array([-x[1], x[0], np.zeros(x.shape[1])])
    solver = NavierStokesSolver(mesh=mesh, markers=tags, params=params, inlet_expression=rotation)
    u, p = solver.simulate()
    assert solver.diagnostics["pressure_gauge"] == "zero_mean"
    if not direct:
        assert solver.diagnostics["linear_solver_options"]["fieldsplit_p_pc_type"] == "jacobi"
    assert abs(integral(mesh, p*ufl.dx)) < 1e-10
    assert solver.diagnostics["convergence"]["pressure_compatibility_residual"] < 1e-9
    assert np.isfinite(u.x.array).all()


def test_analytical_friction_channel():
    mesh, tags = box_case(4)
    params = NavierStokesParameters(nu=.1, wind_speed=1., side_top_boundary="slip", wall_model="friction", beta_wall=.5)
    def inlet(x):
        values = parabolic(x)
        values[0] += .8
        return values
    solver = NavierStokesSolver(mesh=mesh, markers=tags, params=params, inlet_expression=inlet)
    u, p = solver.simulate()
    x = ufl.SpatialCoordinate(mesh)
    exact = ufl.as_vector((4*x[2]*(1-x[2])+.8, 0., 0.))
    assert np.sqrt(integral(mesh, ufl.inner(u-exact, u-exact)*ufl.dx)) < 1e-7
    assert np.sqrt(integral(mesh, (p-.8*(2-x[0]))**2*ufl.dx)) < 1e-7
    assert solver.diagnostics["convergence"]["wall_normal_velocity_rms"] < 1e-10


def test_underresolved_cavity_is_rejected():
    mesh, tags = box_case(3, closed=True)
    solver = NavierStokesSolver(mesh=mesh, markers=tags, params=NavierStokesParameters(nu=.1, closed_cavity=True, side_top_boundary="slip"), inlet_expression=moving_wall)
    with pytest.raises(RuntimeError, match="quality checks failed"):
        solver.simulate()
    assert solver.diagnostics["convergence"]["dimensionless_divergence"] > .1
    assert solver.diagnostics["stop_reason"] == "quality_failure"


@pytest.mark.parametrize("closed", [False, True])
def test_zero_forcing(closed):
    mesh, tags = box_case(2, closed=closed)
    solver = NavierStokesSolver(mesh=mesh, markers=tags, params=NavierStokesParameters(wind_speed=0., closed_cavity=closed, side_top_boundary="slip"))
    u, p = solver.simulate()
    np.testing.assert_array_equal(u.x.array, 0.)
    np.testing.assert_array_equal(p.x.array, 0.)
    assert solver.diagnostics["stop_reason"] == "converged"


def test_closed_flag_with_open_traction_has_no_pressure_nullspace():
    mesh, tags = box_case(2)
    params = NavierStokesParameters(wind_speed=0., closed_cavity=True, side_top_boundary="open")
    solver = NavierStokesSolver(mesh=mesh, markers=tags, params=params)
    solver.simulate()
    assert solver.diagnostics["pressure_gauge"] == "traction_determined"


def test_incompatible_closed_inflow_fails_without_export(tmp_path):
    mesh, tags = box_case(2, closed=True)
    solver = NavierStokesSolver(mesh=mesh, markers=tags, params=NavierStokesParameters(nu=.1, closed_cavity=True), inlet_expression=parabolic, output_dir=tmp_path)
    with pytest.raises(RuntimeError, match="Incompatible prescribed boundary flux"):
        solver.simulate()
    assert not list(tmp_path.glob("*.xdmf"))
    assert solver.diagnostics["stop_reason"] == "solver_failure"


@pytest.mark.parametrize("failure", ["linear", "nonlinear", "residual", "nonfinite"])
def test_failure_preserves_owned_and_unrelated_outputs(failure, tmp_path, monkeypatch):
    mesh, tags = box_case(3, closed=True)
    parameters = dict(nu=.1, closed_cavity=True, side_top_boundary="slip")
    if failure == "linear":
        parameters["petsc_options"] = {"ksp_type": "gmres", "pc_type": "jacobi", "ksp_max_it": 1}
    if failure == "nonlinear":
        parameters["nonlinear_max_it"] = 1
        parameters["nonlinear_atol"] = 1e-14
    if failure in {"residual", "nonfinite"}:
        solve = _MixedFlowSystem.solve
        def corrupted_solve(self, convection):
            result = solve(self, convection)
            if convection == 1.:
                owned = self.W.dofmap.index_map.size_local
                constrained = set(np.concatenate([bc.dof_indices()[0] for bc in self.bcs]))
                free = next(i for i in range(owned) if i not in constrained)
                self.w.x.array[free] = np.nan if failure == "nonfinite" else self.w.x.array[free] + .01
            return result
        monkeypatch.setattr(_MixedFlowSystem, "solve", corrupted_solve)
    outputs = [tmp_path / name for name in ["solution_velocity.xdmf", "solution_velocity.h5", "solution_pressure.xdmf", "solution_pressure.h5", "keep.txt"]]
    for output in outputs:
        output.write_bytes(b"preserve")
    solver = NavierStokesSolver(mesh=mesh, markers=tags, params=NavierStokesParameters(**parameters), inlet_expression=moving_wall, output_dir=tmp_path)
    with pytest.raises(RuntimeError):
        solver.simulate()
    assert all(output.read_bytes() == b"preserve" for output in outputs)
    assert solver.diagnostics["stop_reason"] in {"solver_failure", "quality_failure"}
    assert solver.diagnostics["nonlinear_history"]


def manufactured(n):
    mesh = dm.create_unit_cube(MPI.COMM_WORLD, n, n, n)
    mesh.topology.create_entities(2)
    mesh.topology.create_connectivity(2, 3)
    faces = np.sort(dm.exterior_facet_indices(mesh.topology))
    tags = dm.meshtags(mesh, 2, faces, np.full(len(faces), -1, dtype=np.int32))
    cats = categorize_boundary(mesh, tags, 0, -3, None)
    params = NavierStokesParameters(nu=.1, closed_cavity=True, side_top_boundary="slip")
    system = _MixedFlowSystem(mesh, params, tags, cats, lambda x: np.zeros_like(x))
    x = ufl.SpatialCoordinate(mesh)
    psi = .1*ufl.sin(ufl.pi*x[0])**2*ufl.sin(ufl.pi*x[1])**2*ufl.sin(ufl.pi*x[2])**2
    ue = ufl.as_vector((psi.dx(1), -psi.dx(0), 0.))
    pe = ufl.sin(ufl.pi*x[0])*ufl.sin(ufl.pi*x[1])*ufl.sin(ufl.pi*x[2])
    forcing = -.1*ufl.div(ufl.grad(ue)) + ufl.grad(ue)*ue + ufl.grad(pe)
    v, q = ufl.TestFunctions(system.W)
    system.F -= ufl.inner(forcing, v)*ufl.dx
    system.J = ufl.derivative(system.F, system.w)
    _, dp = ufl.TrialFunctions(system.W)
    system.P = system.J + 10*dp*q*ufl.dx
    from dtcc_sim._flow.solver import DEFAULT_STATIONARY_PETSC
    options = {**DEFAULT_STATIONARY_PETSC, "snes_type": "newtonls", "snes_linesearch_type": "bt", "snes_rtol": 1e-10, "snes_atol": 1e-11, "snes_max_it": 30, "ksp_rtol": 1e-10}
    system.configure(options)
    return system, ue, pe


def test_manufactured_convergence_and_jacobian():
    errors = []
    for n in [2, 4, 8]:
        system, ue, pe = manufactured(n)
        try:
            system.solve(0.)
            system.solve(1.)
            assert system.nullspace is not None
            assert max(system.residual()) < 1e-9
            u, p = system.fields()
            velocity_error = np.sqrt(system.integral(ufl.inner(u-ue, u-ue)*ufl.dx))
            pressure_error = p-pe
            pressure_error -= system.integral(pressure_error*ufl.dx)
            errors.append([velocity_error, np.sqrt(system.integral(pressure_error**2*ufl.dx))])
            if n == 4:
                check_jacobian(system)
        finally:
            system.close()
    errors = np.array(errors)
    fitted_orders = np.log2(errors[0]/errors[2])/2
    finest_orders = np.log2(errors[1]/errors[2])
    assert np.all(fitted_orders > [2.5, 1.5]), (errors, fitted_orders)
    assert np.all(finest_orders > [2.5, 1.5]), (errors, finest_orders)


def check_jacobian(system):
    matrix = fp.assemble_matrix(fem.form(system.J), bcs=system.bcs)
    matrix.assemble()
    direction = system.w.x.petsc_vec.duplicate()
    direction.array[:] = np.random.default_rng(123).normal(size=direction.getLocalSize())
    for bc in system.bcs:
        dofs, owned = bc.dof_indices()
        direction.array[dofs[:owned]] = 0.
    direction.normalize()
    action = direction.duplicate()
    matrix.mult(direction, action)
    original = system.w.x.array.copy()
    residuals = []
    try:
        for sign in [1, -1]:
            system.w.x.array[:] = original
            system.w.x.petsc_vec.axpy(sign*1e-6, direction)
            system.w.x.scatter_forward()
            residual = fp.assemble_vector(fem.form(system.F))
            residual.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            for bc in system.bcs:
                dofs, owned = bc.dof_indices()
                residual.array[dofs[:owned]] = 0.
            residuals.append(residual)
        residuals[0].axpy(-1, residuals[1])
        residuals[0].scale(1/(2e-6))
        residuals[0].axpy(-1, action)
        assert residuals[0].norm()/action.norm() < 1e-6
    finally:
        system.w.x.array[:] = original
        system.w.x.scatter_forward()
        for obj in [matrix, direction, action, *residuals]:
            obj.destroy()


def test_native_export_coordinates_and_replacement(tmp_path):
    from dtcc_core import BoxMesh, io
    source = BoxMesh(3, 3, 3)
    source.vertices += np.array([319000., 6390000., 10.])
    coordinates = source.vertices.copy()
    params = NavierStokesParameters(wind_speed=.1, nu=.1, side_top_boundary="slip",
                                   inlet_profile="power_law", power_law_alpha=1., u_ref_height=1.)
    solver = NavierStokesSolver(mesh=source, params=params, output_dir=tmp_path)
    result = solver.simulate()
    np.testing.assert_array_equal(result.vertices, coordinates)
    target = tmp_path / "fields.dtcc"
    result.save(target)
    restored = io.load_model(target)
    np.testing.assert_array_equal(restored.vertices, coordinates)
    assert {field.name: (field.dim, field.unit, field.association) for field in restored.fields} == {
        "velocity": (3, "m/s", "vertex"), "pressure": (1, "m^2/s^2", "vertex"), "speed": (1, "m/s", "vertex"),
    }
    (tmp_path / "keep.txt").write_text("unrelated")
    (tmp_path / "solution_velocity.xdmf").write_text("replace")
    solver.simulate()
    assert (tmp_path / "keep.txt").read_text() == "unrelated"
    assert "Xdmf" in (tmp_path / "solution_velocity.xdmf").read_text()


def test_uncoupled_pressure_is_rejected_without_regularization(tmp_path):
    mesh = dm.create_mesh(MPI.COMM_WORLD, np.array([[0, 1, 2, 3]], dtype=np.int64),
                          e=ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))),
                          x=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))
    mesh.topology.create_entities(2)
    mesh.topology.create_connectivity(2, 3)
    faces = np.sort(dm.exterior_facet_indices(mesh.topology))
    tags = dm.meshtags(mesh, 2, faces, np.full(len(faces), -3, dtype=np.int32))
    solver = NavierStokesSolver(mesh=mesh, markers=tags, params=NavierStokesParameters(wind_speed=0., closed_cavity=True), output_dir=tmp_path)
    with pytest.raises(RuntimeError, match="uncoupled pressure DOFs"):
        solver.simulate()
    assert not list(tmp_path.glob("*.xdmf"))


def test_iterative_matches_explicit_lu():
    mesh, tags = box_case(3)
    fields = []
    for options in [None, {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}]:
        parameters = NavierStokesParameters(nu=.1, side_top_boundary="slip", petsc_options=options)
        solver = NavierStokesSolver(mesh=mesh, markers=tags, params=parameters, inlet_expression=parabolic)
        u, p = solver.simulate()
        fields.append((u.x.array.copy(), p.x.array.copy()))
    np.testing.assert_allclose(fields[0][0], fields[1][0], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(fields[0][1], fields[1][1], atol=1e-8, rtol=1e-8)


def test_adaptive_linear_solves_keep_native_ew_and_match_lu(monkeypatch):
    mesh, tags = box_case(4)
    matrices = []
    configure = _MixedFlowSystem.configure
    def measured_configure(self, options):
        configure(self, options)
        matrices.append(self.problem.P_mat)
    monkeypatch.setattr(_MixedFlowSystem, "configure", measured_configure)
    def inlet(x):
        result = parabolic(x)
        result[1] = .05*np.sin(np.pi*x[1])*np.sin(np.pi*x[2])
        return result
    fields = []
    histories = []
    for options in [None, {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}]:
        solver = NavierStokesSolver(
            mesh=mesh, markers=tags, inlet_expression=inlet,
            params=NavierStokesParameters(nu=.1, side_top_boundary="slip", petsc_options=options),
        )
        u, p = solver.simulate()
        fields.append((u.x.array.copy(), p.x.array.copy()))
        histories.append(solver.diagnostics["nonlinear_history"])
        assert solver.diagnostics["stop_reason"] == "converged"
    assert matrices == [None, None]
    iterative = histories[0]
    assert all(row["linear_rtol"] == 1e-10 for row in iterative if row["convection"] == 0)
    tolerances = [row["linear_rtol"] for row in iterative if row["convection"] == 1 and row["iteration"] > 0]
    assert tolerances and all(0 < tolerance <= .1 for tolerance in tolerances)
    assert max(tolerances) > 1e-5
    for reference, actual in zip(fields[1], fields[0]):
        np.testing.assert_allclose(actual, reference, rtol=1e-7, atol=1e-9)


def test_refined_native_export_samples_original_vertices():
    from dtcc_core import BoxMesh
    from dtcc_sim.fenics import _mesh_from_dtcc
    from dtcc_sim._flow.solver import _refine_pressure_mesh, _dolfinx_to_volume_mesh
    native = BoxMesh(3, 3, 3)
    mesh, tags = _mesh_from_dtcc(native)
    mesh, tags = _refine_pressure_mesh(mesh, tags, np.array([[0., 0., 0.]]))
    velocity_space = fem.functionspace(mesh, basix.ufl.element("Lagrange", mesh.basix_cell(), 2, shape=(3,)))
    pressure_space = fem.functionspace(mesh, basix.ufl.element("Lagrange", mesh.basix_cell(), 1))
    velocity, pressure = fem.Function(velocity_space), fem.Function(pressure_space)
    def expression(x):
        return np.array([np.sin(3*x[0])*np.cos(4*x[1])+x[2], x[0]**2, x[1]*x[2]])
    velocity.interpolate(expression)
    pressure.interpolate(lambda x: np.sin(5*x[0])+x[1]*x[2])
    result = _dolfinx_to_volume_mesh(mesh, velocity, pressure, native)
    values = {field.name: np.asarray(field.values) for field in result.fields}
    np.testing.assert_allclose(values["velocity"].reshape(-1, 3), expression(native.vertices.T).T, atol=1e-10)
    np.testing.assert_allclose(values["pressure"].ravel(), np.sin(5*native.vertices[:, 0])+native.vertices[:, 1]*native.vertices[:, 2], atol=1e-10)


@pytest.mark.parametrize("scale", [1e-6, 1., 1e6])
@pytest.mark.parametrize("closed", [False, True])
def test_pressure_gauge_is_invariant_to_coordinate_scale(scale, closed):
    mesh, tags = box_case(2, closed=closed)
    mesh.geometry.x[:] *= scale
    solver = NavierStokesSolver(mesh=mesh, markers=tags,
                               params=NavierStokesParameters(wind_speed=0., closed_cavity=closed,
                                                            side_top_boundary="slip"))
    solver.simulate()
    assert solver.diagnostics["pressure_gauge"] == ("zero_mean" if closed else "traction_determined")
