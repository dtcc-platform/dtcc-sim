"""Benchmark enriched stationary flow and exact algebraic bubble elimination."""

from __future__ import annotations

import argparse
import hashlib
import gc
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
from unittest.mock import patch

import basix.ufl
import dolfinx
import dolfinx.fem as fem
from dolfinx.fem import petsc as fp
import dolfinx.geometry as geometry
import dolfinx.mesh as dm
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dtcc_sim._flow import solver as flow
from dtcc_sim import NavierStokesSolver, NavierStokesParameters, StokesSolver, StokesParameters


class BubbleElimination:
    """Solve the full Newton system through its retained Schur complement."""

    def __init__(self, system, timings):
        self.system, self.timings = system, timings
        W, mesh = system.W, system.mesh
        ncells = mesh.topology.index_map(mesh.topology.dim).size_local
        velocity = W.sub(0)
        positions = np.asarray(velocity.ufl_element().entity_dofs[3][0], dtype=np.int32)
        dofmap = velocity.dofmap
        local_bubbles = np.array([dofmap.cell_dofs(c)[positions] for c in range(ncells)], dtype=np.int32)
        self.local_bubbles = local_bubbles.reshape(ncells, 3)
        imap = W.dofmap.index_map
        assert np.all(self.local_bubbles < imap.size_local)
        global_bubbles = imap.local_to_global(self.local_bubbles.ravel())
        self.bubble_is = PETSc.IS().createGeneral(global_bubbles.astype(PETSc.IntType), comm=mesh.comm)
        keep = np.setdiff1d(np.arange(imap.size_local, dtype=np.int32), self.local_bubbles)
        self.retained_is = PETSc.IS().createGeneral(imap.local_to_global(keep).astype(PETSc.IntType), comm=mesh.comm)
        self.objects = []
        self.error = None
        self.S = None
        self.inner = PETSc.KSP().create(mesh.comm)
        self.inner.setOptionsPrefix('dtcc_bubble_inner_')
        self.inner.setType('preonly')
        self.inner.getPC().setType('lu')
        self.inner.getPC().setFactorSolverType('mumps')
        opts = PETSc.Options()
        for key, value in {'mat_mumps_icntl_24': 1, 'mat_mumps_icntl_25': 0}.items():
            opts[self.inner.getOptionsPrefix()+key] = value
        self.inner.setFromOptions()
        for key in ['mat_mumps_icntl_24', 'mat_mumps_icntl_25']:
            del opts[self.inner.getOptionsPrefix()+key]

    def setUp(self, pc):
        self.error = None
        try:
            self._set_up(pc)
        except Exception as error:
            self.error = str(error)
            raise

    def _set_up(self, pc):
        for obj in reversed(self.objects):
            obj.destroy()
        self.objects = []
        started = time.perf_counter()
        A, _ = pc.getOperators()
        ir, ib = self.retained_is, self.bubble_is
        self.Arr = A.createSubMatrix(ir, ir)
        self.Arb = A.createSubMatrix(ir, ib)
        self.Abr = A.createSubMatrix(ib, ir)
        Abb = A.createSubMatrix(ib, ib)
        self.objects.extend([self.Arr, self.Arb, self.Abr, Abb])
        lo, hi = Abb.getOwnershipRange()
        blocks = np.zeros((len(self.local_bubbles), 3, 3))
        indptr, indices, values = Abb.getValuesCSR()
        rows = np.repeat(np.arange(hi-lo), np.diff(indptr))
        same = indices//3 == (lo+rows)//3
        if np.any(abs(values[~same]) > 1e-12*max(np.max(abs(values)), 1.)):
            raise RuntimeError('Bubble block is not cell local')
        blocks[rows[same]//3, rows[same]%3, indices[same]%3] = values[same]
        inverses = np.linalg.inv(blocks)
        if not np.isfinite(inverses).all():
            raise RuntimeError('Nonfinite bubble inverse')
        columns = np.repeat(np.arange(lo, hi, 3, dtype=PETSc.IntType), 9)
        columns += np.tile(np.arange(3, dtype=PETSc.IntType), 3*len(blocks))
        inverse = PETSc.Mat().createAIJ(size=Abb.getSizes(), csr=(np.arange(0, 3*(hi-lo)+1, 3, dtype=PETSc.IntType), columns, inverses.ravel()), comm=A.comm)
        self.inverse = inverse
        self.invAbr = inverse.matMult(self.Abr)
        correction = self.Arb.matMult(self.invAbr)
        condensed = self.Arr.copy()
        condensed.axpy(-1., correction, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        condensed.assemble()
        if self.S is None:
            self.S = condensed
        else:
            condensed.copy(result=self.S, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
            condensed.destroy()
        self.objects.extend([inverse, self.invAbr, correction])
        self.fb, self.xb = Abb.createVecs()
        self.fr, self.xr = self.S.createVecs()
        self.tb = self.fb.duplicate()
        self.tr = self.fr.duplicate()
        self.objects.extend([self.fb, self.xb, self.fr, self.xr, self.tb, self.tr])
        if self.system.nullspace is not None:
            full = self.system.null_vector
            restricted = full.getSubVector(ir)
            gauge = restricted.copy()
            full.restoreSubVector(ir, restricted)
            gauge.normalize()
            null = PETSc.NullSpace().create(vectors=[gauge], comm=A.comm)
            self.S.setNullSpace(null)
            self.S.setTransposeNullSpace(null)
            self.objects.extend([gauge, null])
        self.timings['condensation_s'] += time.perf_counter()-started
        started = time.perf_counter()
        self.inner.setOperators(self.S)
        self.inner.setUp()
        self.timings['condensed_factor_setup_s'] += time.perf_counter()-started
        self.timings['condensation_count'] += 1

    def apply(self, pc, rhs, result):
        try:
            self._apply(pc, rhs, result)
        except Exception as error:
            self.error = str(error)
            raise

    def _apply(self, pc, rhs, result):
        started = time.perf_counter()
        r = rhs.getSubVector(self.retained_is)
        b = rhs.getSubVector(self.bubble_is)
        r.copy(self.fr)
        b.copy(self.fb)
        rhs.restoreSubVector(self.retained_is, r)
        rhs.restoreSubVector(self.bubble_is, b)
        self.inverse.mult(self.fb, self.tb)
        self.Arb.mult(self.tb, self.tr)
        self.fr.axpy(-1., self.tr)
        self.timings['condensed_rhs_s'] += time.perf_counter()-started
        started = time.perf_counter()
        self.inner.solve(self.fr, self.xr)
        self.timings['condensed_triangular_s'] += time.perf_counter()-started
        if self.inner.getConvergedReason() <= 0:
            raise RuntimeError(f'Condensed linear solve failed: {self.inner.getConvergedReason()}')
        started = time.perf_counter()
        self.Abr.mult(self.xr, self.tb)
        self.tb.aypx(-1., self.fb)
        self.inverse.mult(self.tb, self.xb)
        r = result.getSubVector(self.retained_is)
        b = result.getSubVector(self.bubble_is)
        self.xr.copy(r)
        self.xb.copy(b)
        result.restoreSubVector(self.retained_is, r)
        result.restoreSubVector(self.bubble_is, b)
        self.timings['bubble_recovery_s'] += time.perf_counter()-started
        self.timings['retained_dofs'] = self.S.getSize()[0]

    def destroy(self, pc=None):
        self.inner.destroy()
        if self.S is not None:
            self.S.destroy()
            self.S = None
        for obj in reversed(self.objects):
            obj.destroy()
        self.objects = []
        self.bubble_is.destroy()
        self.retained_is.destroy()


def profiled_system(element_name, condensed, record):
    base = flow._MixedFlowSystem
    native_element = basix.ufl.element
    class ProfiledSystem(base):
        last_fields = None

        def __init__(self, mesh, *args):
            started = time.perf_counter()
            def selected_element(family, cell, degree, **kwargs):
                if family == 'Lagrange' and degree == 2 and 'shape' in kwargs and element_name != 'p2':
                    scalar = basix.ufl.enriched_element([
                        native_element('Lagrange', cell, 1 if element_name == 'mini' else 2),
                        native_element('Bubble', cell, 4),
                    ])
                    return basix.ufl.blocked_element(scalar, shape=kwargs['shape'])
                return native_element(family, cell, degree, **kwargs)
            with patch.object(basix.ufl, 'element', selected_element):
                super().__init__(mesh, *args)
            record['system_creation_s'] += time.perf_counter()-started
            self.elimination = None
            self.metrics = record

        def configure(self, options):
            started = time.perf_counter()
            if condensed:
                options = {key:value for key,value in options.items() if not key.startswith(('fieldsplit_', 'pc_fieldsplit_', 'ksp_'))}
                options.update(ksp_type='preonly', pc_type='python')
                options.pop('pc_factor_mat_solver_type', None)
            super().configure(options)
            record['configuration_s'] += time.perf_counter()-started
            problem = self.problem
            def residual(snes, x, b):
                started = time.perf_counter()
                fp.assemble_residual(snes, x, b, self.w, problem.F, problem.J, self.bcs)
                if self.nullspace:
                    self.nullspace.remove(b)
                record['residual_assembly_s'] += time.perf_counter()-started
                record['residual_count'] += 1
            def jacobian(snes, x, A, P):
                started = time.perf_counter()
                fp.assemble_jacobian(snes, x, A, P, self.w, problem.J, problem.preconditioner, self.bcs)
                record['jacobian_assembly_s'] += time.perf_counter()-started
                record['jacobian_count'] += 1
            problem.solver.setFunction(residual, problem.b)
            problem.solver.setJacobian(jacobian, problem.A, problem.P_mat)
            ksp = problem.solver.getKSP()
            def presolve(ksp, b, x):
                self.linear_started = time.perf_counter()
            def postsolve(ksp, b, x):
                record['linear_total_s'] += time.perf_counter()-self.linear_started
                record['linear_count'] += 1
            ksp.setPreSolve(presolve)
            ksp.setPostSolve(postsolve)
            if condensed:
                self.elimination = BubbleElimination(self, record)
                ksp.getPC().setPythonContext(self.elimination)
            record['full_dofs'] = self.W.dofmap.index_map.size_global

        def solve(self, convection):
            try:
                return super().solve(convection)
            except Exception as error:
                if self.elimination is not None and self.elimination.error:
                    raise RuntimeError(f'Bubble elimination failed: {self.elimination.error}') from error
                raise

        def fields(self):
            fields = super().fields()
            type(self).last_fields = fields
            return fields

        def close(self):
            if self.problem is not None:
                problem = self.problem
                problem.solver.setJacobian(fp.assemble_jacobian, problem.A, problem.P_mat, kargs=dict(u=self.w, jacobian=problem.J, preconditioner=problem.preconditioner, bcs=self.bcs))
                self.problem.solver.getKSP().setPreSolve(None)
                self.problem.solver.getKSP().setPostSolve(None)
                if self.elimination is not None:
                    self.problem.solver.getKSP().getPC().setPythonContext(None)
                    self.elimination.destroy()
                    self.elimination = None
            super().close()
    return ProfiledSystem


def case_mesh(case, n, repaired, source, uniform_refinements=0):
    if case == 'city':
        mesh, tags = flow.load_mesh_with_markers(str(source/'city.xdmf'))
        flow.offset_to_origin(mesh)
        for _ in range(uniform_refinements):
            mesh.topology.create_entities(1)
            refined, parents, parent_facets = dm.refine(mesh, option=dm.RefinementOption.parent_cell_and_facet)
            refined.topology.create_entities(2)
            refined.topology.create_connectivity(2,3)
            tags = dm.transfer_meshtag(tags,refined,parents,parent_facets)
            mesh = refined
        physical = dict(wind_speed=5., wind_dir_deg=270., nu_t=2., side_top_boundary='slip', inlet_profile='log_law', grad_div_gamma=3., backflow_beta=2.)
        if repaired:
            params = NavierStokesParameters(**physical)
            categories = flow.categorize_boundary(mesh,tags,flow.infer_num_buildings(mesh,tags),-3,-4)
            probe = flow._MixedFlowSystem(mesh,params,tags,categories,flow.make_inlet_velocity_expression(params))
            points = None
            try:
                probe.configure(dict(ksp_type='preonly',pc_type='lu',pc_factor_mat_solver_type='mumps'))
            except flow._PressureSpaceError as error:
                points = error.coordinates
            finally:
                probe.close()
            if points is not None:
                mesh,tags = flow._refine_pressure_mesh(mesh,tags,points)
        inlet = None
    else:
        mesh = dm.create_box(MPI.COMM_WORLD, [[0.,0.,0.],[2.,1.,1.]], [2*n,n,n])
        mesh.topology.create_entities(2)
        mesh.topology.create_connectivity(2,3)
        faces = dm.exterior_facet_indices(mesh.topology)
        mid = dm.compute_midpoints(mesh,2,faces)
        values = np.full(len(faces),-1,dtype=np.int32)
        for axis, coordinate, marker in [(0,0,-3),(0,2,-4),(1,0,-5),(1,1,-6)]:
            values[np.isclose(mid[:,axis],coordinate)] = marker
        order = np.argsort(faces)
        tags = dm.meshtags(mesh,2,faces[order],values[order])
        physical = dict(wind_speed=1., nu=.1, side_top_boundary='slip', divergence_tolerance=.005)
        def inlet(x):
            return np.array([4*x[2]*(1-x[2]), np.zeros(x.shape[1]), np.zeros(x.shape[1])])
    return mesh, tags, physical, inlet


def sample(mesh, u, p):
    x,y=np.meshgrid(np.linspace(2,198,50), np.linspace(2,198,50))
    points=np.concatenate([np.column_stack((x.ravel(),y.ravel(),np.full(x.size,z))) for z in [10.,30.]])
    tree=geometry.bb_tree(mesh,3)
    candidates=geometry.compute_collisions_points(tree,points)
    collisions=geometry.compute_colliding_cells(mesh,candidates,points)
    cells=np.array([collisions.links(i)[0] if len(collisions.links(i)) else -1 for i in range(len(points))],dtype=np.int32)
    valid=cells>=0
    uv=np.full((len(points),3),np.nan);pv=np.full(len(points),np.nan)
    uv[valid]=u.eval(points[valid],cells[valid]);pv[valid]=p.eval(points[valid],cells[valid]).ravel()
    return dict(points=points,u=uv,p=pv)


def empty_metrics():
    record={key:0. for key in ['system_creation_s','configuration_s','residual_assembly_s','jacobian_assembly_s','linear_total_s','condensation_s','condensed_factor_setup_s','condensed_rhs_s','condensed_triangular_s','bubble_recovery_s']}
    record.update({key:0 for key in ['residual_count','jacobian_count','linear_count','condensation_count']})
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--element', choices=['p2','p2b','mini'], required=True)
    parser.add_argument('--condense', action='store_true')
    parser.add_argument('--case', choices=['channel','city'], default='channel')
    parser.add_argument('--repaired', action='store_true')
    parser.add_argument('--uniform-refinements', type=int, default=0)
    parser.add_argument('--n', type=int, default=4)
    parser.add_argument('--runs', type=int, default=4)
    parser.add_argument('--source', type=Path, default=ROOT/'.agent/work/stationary')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cold-cache', type=Path)
    args=parser.parse_args()
    if args.condense and args.element == 'p2':
        parser.error('P2 has no cell bubbles to condense')
    args.output.mkdir(parents=True, exist_ok=True)
    PETSc.Log.begin()
    if args.cold_cache:
        dolfinx.jit.DOLFINX_DEFAULT_JIT_OPTIONS['cache_dir']=(str(args.cold_cache),'cache')
    jit_seconds=0.
    original_jit=dolfinx.jit.ffcx_jit
    def measured_jit(*a, **kw):
        nonlocal jit_seconds
        started=time.perf_counter()
        try:
            return original_jit(*a, **kw)
        finally:
            jit_seconds += time.perf_counter()-started
    records=[]
    preparation_started=time.perf_counter()
    mesh,tags,physical,inlet=case_mesh(args.case,args.n,args.repaired,args.source,args.uniform_refinements)
    preparation_s=time.perf_counter()-preparation_started
    coords=mesh.geometry.x.copy()
    checksum=hashlib.sha256(coords.tobytes()+mesh.geometry.dofmaps[0].tobytes()+tags.values.tobytes()).hexdigest()
    for run in range(args.runs):
        jit_seconds=0.
        record=empty_metrics()
        record['run']=run
        solver_type=profiled_system(args.element,args.condense,record)
        pc=dict(ksp_type='preonly',pc_type='python' if args.condense else 'lu')
        if not args.condense:
            pc['pc_factor_mat_solver_type']='mumps'
        params=NavierStokesParameters(**physical,petsc_options=pc)
        mesh.geometry.x[:]=coords
        solver=NavierStokesSolver(mesh=mesh,markers=tags,params=params,inlet_expression=inlet)
        stage=PETSc.Log.Stage(f'flow_run_{run}')
        stage.push()
        started=time.perf_counter()
        cpu_started=time.process_time()
        try:
            with patch.object(flow,'_MixedFlowSystem',solver_type), patch.object(dolfinx.jit,'ffcx_jit',measured_jit):
                u,p=solver.simulate()
            record['wall_s']=time.perf_counter()-started
            record['cpu_s']=time.process_time()-cpu_started
            record['jit_s']=jit_seconds
            record['diagnostics']=solver.diagnostics
            record['accepted']=True
            np.savez(args.output/f'fields-{run}.npz',u=u.x.array,p=p.x.array)
            if args.case == 'city' and mesh.comm.size == 1:
                np.savez(args.output/f'probes-{run}.npz',**sample(mesh,u,p))
        except Exception as exc:
            record['cpu_s']=time.process_time()-cpu_started
            record.update(wall_s=time.perf_counter()-started,jit_s=jit_seconds,accepted=False,error=repr(exc),traceback=traceback.format_exc(),diagnostics=solver.diagnostics)
        finally:
            record['petsc_events']={event:PETSc.Log.Event(event).getPerfInfo(stage.id) for event in ['MatLUFactorSym','MatLUFactorNum','MatSolve','KSPSolve','PCSetUp','MatMult']}
            stage.pop()
        if args.case == 'channel' and solver_type.last_fields is not None:
            u,p=solver_type.last_fields
            x=ufl.SpatialCoordinate(mesh)
            ue=ufl.as_vector((4*x[2]*(1-x[2]),0.,0.))
            record['errors']={key:float(np.sqrt(mesh.comm.allreduce(fem.assemble_scalar(fem.form(expr)),op=MPI.SUM))) for key,expr in {'velocity_l2':ufl.inner(u-ue,u-ue)*ufl.dx,'pressure_l2':(p-.8*(2-x[0]))**2*ufl.dx}.items()}
            record['reference_accuracy_pass']=all(v<.002 for v in record['errors'].values())
        record['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        record['all_gates_pass'] = record['accepted'] and record.get('reference_accuracy_pass', True) and solver.diagnostics['convergence']['wall_normal_velocity_rms'] <= 1e-8
        records.append(record)
        if mesh.comm.rank == 0:
            manifest=dict(uniform_refinements=args.uniform_refinements,case=args.case,element=args.element,condensed=args.condense,repaired=args.repaired,n=args.n,cells=mesh.topology.index_map(3).size_global,mesh_sha256=checksum,preparation_s=preparation_s,parameters=params.model_dump(),dolfinx=dolfinx.__version__,petsc=PETSc.Sys.getVersion(),python=sys.version,hardware=platform.uname()._asdict(),threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},source_sha256=hashlib.sha256((ROOT/'dtcc_sim/_flow/solver.py').read_bytes()).hexdigest(),records=records)
            (args.output/'measurements.json').write_text(json.dumps(manifest,indent=2))
            print(json.dumps(record),flush=True)
        gc.collect()


if __name__ == '__main__':
    main()
