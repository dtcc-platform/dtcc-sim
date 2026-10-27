"""Exercise actual mesh generation, both solvers and exports in a fresh prefix."""
from pathlib import Path
import importlib, json, os, sys, time
R=None
PREFIX=Path(sys.prefix).resolve()

def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')

def boundaries(mesh):
    markers=mesh.boundary_markers
    if isinstance(markers,dict):return {tuple(sorted(k)):int(v) for k,v in markers.items()}
    return {tuple(sorted(map(int,k))):int(v) for k,v in zip(mesh.boundary_faces,markers)}

def main():
    global R
    R=Path(sys.argv[1]).resolve()
    R.mkdir(parents=True, exist_ok=False)
    begin=time.monotonic()
    assert Path(sys.prefix).resolve()==PREFIX.resolve()
    assert not os.environ.get('PYTHONPATH') and os.environ.get('PYTHONNOUSERSITE')=='1'
    modules={}
    for name in ['dtcc_core','dtcc_sim','numpy','scipy','dolfinx','h5py','pymetis','dtcc_mesher','dtcc_tetgen_wrapper','dtcc_core.builder._dtcc_builder']:
        module=importlib.import_module(name);path=Path(module.__file__).resolve()
        assert path.is_relative_to(PREFIX.resolve()),(name,str(path))
        modules[name]=str(path)
    import numpy as np
    import meshio
    from scipy.sparse import csr_matrix
    from dtcc_core import datasets
    import dtcc_sim.datasets
    from dtcc_core.io import load_volume_mesh
    from dtcc_core.datasets import load_model_package
    from dtcc_core.datasets.package import export_model_package
    from dtcc_sim.helmholtz import HelmholtzResult,HelmholtzParameters,HelmholtzSimulator
    from dtcc_sim.helmholtz.simulator import _configured_backend
    from dtcc_sim.helmholtz.check_runtime import inspect_runtime
    readiness=inspect_runtime(workers=2,mesh_generation=True);assert readiness['ready']
    write(R/'readiness.json',readiness)
    rows={}
    for mode in ['geometric_hats','core_overlap_ramp']:
        directory=R/('case-'+mode);start=time.monotonic()
        args={'h':25.0,'bounds':[319945.962899,6398959.716755,320045.962899,6399059.716755],
            'domain_height':40.0,'min_building_detail':1.0,'mesh_options':{'flat_ground':True,'show_footprints':False},
            'strict_live':False,'mesh_cache_dir':R/'mesh-cache','output_dir':directory,
            'parameters':HelmholtzParameters(frequency_hz=5.0,rho=1.1,n_subdomains=2,overlap_layers=1,oversample_layers=1,workers=2,threads_per_worker=1,pou_mode=mode)}
        mesh=datasets.helmholtz_simulation(**args)
        result=HelmholtzResult.load(directory)
        assert result.converged and result.relative_residual<=1e-5 and np.isfinite(result.solution).all()
        assert result.prepared_identity['parameters']['pou_mode']==mode
        sim=HelmholtzSimulator(mesh_path=result.mesh_path,output_dir=directory,parameters=args['parameters'])
        with _configured_backend(sim._params(),1):
            from dtcc_sim.helmholtz._backend.mesh import build_system
            assembled=build_system();matrix,rhs=assembled[-2:]
            ptr,col,data=matrix.getValuesCSR();A=csr_matrix((data.copy(),col.copy(),ptr.copy()),shape=matrix.getSize())
            b=rhs.getArray(readonly=True).copy()
            residual=float(np.linalg.norm(A@result.solution-b)/np.linalg.norm(b))
        assert residual<=1e-5 and np.isclose(residual,result.relative_residual,rtol=1e-8,atol=1e-13)
        for suffix in ['dtcc','xdmf']:
            p=directory/('field.'+suffix);mesh.save(p);restored=load_volume_mesh(p)
            np.testing.assert_array_equal(mesh.vertices,restored.vertices)
            np.testing.assert_array_equal(mesh.cells,restored.cells)
            assert boundaries(mesh)==boundaries(restored)
            expected={f.name:f.values for f in mesh.fields}
            assert set(expected)=={f.name for f in restored.fields}
            for f in restored.fields:np.testing.assert_array_equal(expected[f.name],f.values)
        mesh.save(directory/'field.vtu');vtk=meshio.read(directory/'field.vtu')
        np.testing.assert_array_equal(vtk.points[:,:3],mesh.vertices)
        np.testing.assert_array_equal(vtk.cells_dict['tetra'],mesh.cells)
        for f in mesh.fields:np.testing.assert_array_equal(vtk.point_data[f.name].ravel(),f.values.ravel())
        export_model_package(mesh,directory/'field.dtccpkg',canonical=True)
        restored=load_model_package(directory/'field.dtccpkg')
        np.testing.assert_array_equal(mesh.vertices,restored.vertices)
        np.testing.assert_array_equal(mesh.cells,restored.cells)
        assert boundaries(mesh)==boundaries(restored)
        assert mesh.dataset_context.model_dump(mode='json')==restored.dataset_context.model_dump(mode='json')
        fields={f.name:f.values for f in mesh.fields}
        for f in restored.fields:np.testing.assert_array_equal(fields[f.name],f.values)
        provenance=json.loads((directory/'mesh_provenance.json').read_text())
        assert bool(provenance['reused'])==(mode=='core_overlap_ramp')
        rows[mode]={'passed':True,'iterations':result.iterations,'independent_true_residual':residual,
            'vertices':len(mesh.vertices),'tetrahedra':len(mesh.cells),'boundary_facets':len(boundaries(mesh)),
            'mesh_generated':not provenance['reused'],'exports_verified':['dtcc','xdmf','vtu','dtccpkg'],
            'geometry_fields_boundaries_and_package_context_exact':True,'seconds':time.monotonic()-start}
        write(R/'smoke-progress.json',rows)
    report={'passed':True,'clean_prefix':str(PREFIX),'loaded_modules':modules,'cases':rows,
        'no_previous_overlay_or_site_packages_used':True,'seconds':time.monotonic()-begin,
        'scope':'Fresh copied Conda package installation plus pinned Python dependencies; rebuilt project native packages and reviewed Sim wheel. Same Linux host and cached upstream geodata, not a new machine or historical performance benchmark.'}
    write(R/'smoke-validation.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
