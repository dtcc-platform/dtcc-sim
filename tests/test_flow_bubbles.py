"""Numerical accuracy and exact elimination checks for enriched flow elements."""

from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip('dolfinx')
from petsc4py import PETSc
import ufl

from dtcc_sim._flow import solver as flow
from dtcc_sim import NavierStokesSolver, NavierStokesParameters, StokesSolver, StokesParameters
from tools.benchmarks.flow_bubbles import empty_metrics, profiled_system
import test_stationary_flow as reference

pytestmark = [pytest.mark.fenics, pytest.mark.simulation, pytest.mark.slow]


@pytest.fixture(autouse=True)
def real_runtime():
    if not np.issubdtype(PETSc.ScalarType, np.floating):
        pytest.skip('Flow enrichment requires real PETSc')


def manufactured(n, element, condensed=False, direct=True):
    system_type = profiled_system(element, condensed, empty_metrics())
    options = dict(ksp_type='preonly', pc_type='lu', pc_factor_mat_solver_type='mumps', mat_mumps_icntl_24=1, mat_mumps_icntl_25=0, snes_rtol=1e-12, snes_atol=1e-13)
    if direct:
        with patch.object(reference, '_MixedFlowSystem', system_type), patch.object(flow, 'DEFAULT_STATIONARY_PETSC', options):
            return reference.manufactured(n)
    with patch.object(reference, '_MixedFlowSystem', system_type):
        return reference.manufactured(n)


@pytest.mark.parametrize('element', ['p2b', 'mini'])
def test_enriched_manufactured_convergence(element):
    errors = []
    for n in ([2, 4, 8] if element == 'p2b' else [4, 8, 16]):
        system, ue, pe = manufactured(n, element, direct=False)
        try:
            system.solve(0.)
            system.solve(1.)
            assert system.nullspace is not None
            assert max(system.residual()) < 1e-9
            u, p = system.fields()
            ep = p-pe
            ep -= system.integral(ep*ufl.dx)
            errors.append([np.sqrt(system.integral(ufl.inner(u-ue, u-ue)*ufl.dx)), np.sqrt(system.integral(ep**2*ufl.dx))])
            if n == 4:
                reference.check_jacobian(system)
        finally:
            system.close()
    errors = np.asarray(errors)
    print(element, 'manufactured errors', errors.tolist(), 'orders', (np.log2(errors[1]/errors[2])).tolist())
    minimum = [2.5, 1.5] if element == 'p2b' else [1.8, .8]
    assert np.all(np.log2(errors[0]/errors[2])/2 > minimum), errors
    assert np.all(np.log2(errors[1]/errors[2]) > minimum), errors


@pytest.mark.parametrize('element', ['p2b', 'mini'])
def test_condensed_newton_correction_and_nonlinear_field(element):
    fields = []
    corrections = []
    for condensed in [False, True]:
        system, _, _ = manufactured(3, element, condensed)
        try:
            system.convection.value = PETSc.ScalarType(1.)
            system.w.x.array[:] = np.random.default_rng(981).normal(scale=.01, size=len(system.w.x.array))
            flow._fem_petsc.set_bc(system.w.x.petsc_vec, system.bcs)
            system.w.x.scatter_forward()
            problem = system.problem
            flow._fem_petsc.assign(system.w, problem.x)
            problem.solver.computeFunction(problem.x, problem.b)
            problem.solver.computeJacobian(problem.x, problem.A)
            direction = problem.b.duplicate()
            ksp = problem.solver.getKSP()
            ksp.setOperators(problem.A)
            ksp.solve(problem.b, direction)
            assert ksp.getConvergedReason() > 0
            defect = problem.b.duplicate()
            problem.A.mult(direction, defect)
            defect.axpy(-1., problem.b)
            assert defect.norm()/problem.b.norm() < 1e-9
            corrections.append(direction.array.copy())
            direction.destroy()
            defect.destroy()
            system.w.x.array[:] = 0.
            system.solve(0.)
            system.solve(1.)
            assert max(system.residual()) < 1e-9
            u,p = system.fields()
            fields.append((u.x.array.copy(),p.x.array.copy()))
        finally:
            system.close()
    np.testing.assert_allclose(corrections[0], corrections[1], rtol=1e-8, atol=1e-9)
    for a,b in zip(fields[0],fields[1]):
        np.testing.assert_allclose(a,b,rtol=1e-8,atol=1e-9)


@pytest.mark.parametrize('element', ['p2b', 'mini'])
@pytest.mark.parametrize('condensed', [False, True])
def test_corner_tetrahedra_recover_linear_pressure(element, condensed):
    import basix.ufl
    import dolfinx.fem as fem
    import dolfinx.mesh as dm
    from mpi4py import MPI
    points = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.], [0.,0.,-1.]])
    cells = np.array([[0,1,2,3], [0,2,1,4]], dtype=np.int64)
    coordinate = ufl.Mesh(basix.ufl.element('Lagrange', 'tetrahedron', 1, shape=(3,)))
    mesh = dm.create_mesh(MPI.COMM_WORLD, cells, coordinate, points)
    mesh.topology.create_connectivity(2,3)
    facets = np.sort(dm.exterior_facet_indices(mesh.topology))
    tags = dm.meshtags(mesh, 2, facets, np.full(len(facets), -1, dtype=np.int32))
    categories = flow.categorize_boundary(mesh, tags, 0, -3, None)
    params = NavierStokesParameters(nu=.1, closed_cavity=True)
    options = dict(ksp_type='preonly', pc_type='lu', pc_factor_mat_solver_type='mumps', mat_mumps_icntl_24=1, mat_mumps_icntl_25=0, snes_rtol=1e-12, snes_atol=1e-13)
    baseline = flow._MixedFlowSystem(mesh, params, tags, categories, lambda x: np.zeros_like(x))
    try:
        with pytest.raises(flow._PressureSpaceError):
            baseline.configure(options)
    finally:
        baseline.close()
    system_type = profiled_system(element, condensed, empty_metrics())
    system = system_type(mesh, params, tags, categories, lambda x: np.zeros_like(x))
    try:
        x = ufl.SpatialCoordinate(mesh)
        exact = x[0]+2*x[1]+3*x[2]
        v, _ = ufl.TestFunctions(system.W)
        system.F -= ufl.inner(ufl.grad(exact),v)*ufl.dx
        system.J = ufl.derivative(system.F,system.w)
        system.configure(options)
        system.solve(0.)
        u,p = system.fields()
        volume=system.integral(fem.Constant(mesh,PETSc.ScalarType(1.))*ufl.dx)
        exact -= system.integral(exact*ufl.dx)/volume
        assert system.nullspace is not None
        assert max(system.residual()) < 1e-10
        assert np.sqrt(system.integral(ufl.inner(u,u)*ufl.dx)) < 1e-10
        assert np.sqrt(system.integral((p-exact)**2*ufl.dx)) < 1e-10
    finally:
        system.close()


def test_condensation_failure_preserves_outputs(tmp_path, monkeypatch):
    from tools.benchmarks.flow_bubbles import BubbleElimination
    mesh,tags=reference.box_case(2)
    system_type=profiled_system('mini', True, empty_metrics())
    monkeypatch.setattr(flow,'_MixedFlowSystem',system_type)
    def failed_setup(self,pc):
        raise RuntimeError('Singular cell bubble block')
    monkeypatch.setattr(BubbleElimination,'_set_up',failed_setup)
    outputs=[tmp_path/name for name in ['solution_velocity.xdmf','solution_velocity.h5','solution_pressure.xdmf','solution_pressure.h5','keep.txt']]
    for output in outputs:
        output.write_bytes(b'preserve')
    solver=NavierStokesSolver(mesh=mesh,markers=tags,params=NavierStokesParameters(nu=.1),inlet_expression=reference.parabolic,output_dir=tmp_path)
    with pytest.raises(RuntimeError,match='Singular cell bubble block'):
        solver.simulate()
    assert 'Singular cell bubble block' in solver.diagnostics['error']
    assert all(output.read_bytes()==b'preserve' for output in outputs)
    assert solver.diagnostics['stop_reason']=='solver_failure'


@pytest.mark.parametrize('element', ['p2b', 'mini'])
def test_condensed_native_export_and_replacement(element, tmp_path, monkeypatch):
    from dtcc_core import BoxMesh, io
    system_type=profiled_system(element, True, empty_metrics())
    monkeypatch.setattr(flow,'_MixedFlowSystem',system_type)
    n=6 if element=='mini' else 3
    source=BoxMesh(n,n,n)
    source.vertices += np.array([319000.,6390000.,10.])
    coordinates=source.vertices.copy()
    params=NavierStokesParameters(wind_speed=.1,nu=.1,side_top_boundary='slip',inlet_profile='power_law',power_law_alpha=1.,u_ref_height=1.)
    solver=NavierStokesSolver(mesh=source,params=params,output_dir=tmp_path)
    result=solver.simulate()
    np.testing.assert_array_equal(result.vertices,coordinates)
    target=tmp_path/'fields.dtcc'
    result.save(target)
    restored=io.load_model(target)
    np.testing.assert_array_equal(restored.vertices,coordinates)
    assert {field.name:(field.dim,field.unit,field.association) for field in restored.fields} == {
        'velocity':(3,'m/s','vertex'), 'pressure':(1,'m^2/s^2','vertex'), 'speed':(1,'m/s','vertex')}
    (tmp_path/'keep.txt').write_text('unrelated')
    (tmp_path/'solution_velocity.xdmf').write_text('replace')
    solver.simulate()
    assert (tmp_path/'keep.txt').read_text()=='unrelated'
    assert 'Xdmf' in (tmp_path/'solution_velocity.xdmf').read_text()


def test_underresolved_native_mini_is_rejected(tmp_path,monkeypatch):
    system_type=profiled_system('mini',True,empty_metrics())
    monkeypatch.setattr(flow,'_MixedFlowSystem',system_type)
    with pytest.raises(RuntimeError,match='quality checks failed'):
        reference.test_native_export_coordinates_and_replacement(tmp_path)
    assert not list(tmp_path.glob('*.xdmf'))
