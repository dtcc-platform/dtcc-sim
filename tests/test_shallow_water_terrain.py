"""Measured terrain sampling and shallow-water bed/context consistency."""

import numpy as np
import pytest
from affine import Affine
from dtcc_core.model import Raster

from dtcc_sim.shallow_water._terrain import _sample_dem, _city_with_solver_ground


@pytest.mark.parametrize('flip_x,flip_y', [(False,False),(True,False),(False,True),(True,True)])
def test_dem_samples_pixel_centers_with_clamped_edge_margins(flip_x, flip_y):
    raster = Raster(data=np.array([[11., 13.], [17., 19.]]), georef=Affine(2.,0.,319845.,0.,-2.,6397789.))
    xy=np.array([[319846.,6397788.],[319847.,6397787.],[319845.,6397789.],
                 [319849.,6397785.],[319846.,6397785.]])
    if flip_x:
        raster.data=raster.data[:,::-1]
        raster.georef=raster.georef*Affine.translation(2.,0.)*Affine.scale(-1.,1.)
    if flip_y:
        raster.data=raster.data[::-1]
        raster.georef=raster.georef*Affine.translation(0.,2.)*Affine.scale(1.,-1.)
    np.testing.assert_allclose(_sample_dem(raster,xy),[11.,15.,11.,19.,17.],rtol=0,atol=1e-12)
    with pytest.raises(ValueError,match='coverage'):
        _sample_dem(raster,[[319844.,6397788.]])


@pytest.mark.parametrize('fault',['nan','nodata','masked','rotated','zero_spacing'])
def test_invalid_dem_fails_instead_of_inventing_elevations(fault):
    raster=Raster(data=np.ones((2,2)),georef=Affine(1.,0.,0.,0.,-1.,2.))
    if fault=='nan':raster.data[0,0]=np.nan
    elif fault=='nodata':raster.nodata=1.
    elif fault=='masked':raster.data=np.ma.array(raster.data,mask=[[True,False],[False,False]])
    elif fault=='rotated':raster.georef=Affine(1.,.1,0.,0.,-1.,2.)
    else:raster.georef=Affine(0.,0.,0.,0.,-1.,2.)
    with pytest.raises(ValueError,match='DEM'):
        _sample_dem(raster,[[.5,.5]])


def test_city_context_reuses_exact_solver_ground_and_keeps_buildings():
    from dtcc_core.model import Mesh
    mesh=Mesh(vertices=np.array([[0.,0.,10.],[1.,0.,11.],[0.,1.,12.],[1.,1.,13.]]),
              faces=np.array([[0,1,2],[1,2,3]]),markers=np.array([-2,0]))
    mesh.transform.srs='EPSG:3006'
    surface=Mesh(vertices=np.array([[0.,0.,8.],[1.,0.,8.],[0.,1.,8.],[0.,0.,20.]]),
                 faces=np.array([[0,1,2],[0,1,3]],dtype=np.uint64),markers=np.array([-2,0]))
    original=surface.vertices.copy()
    context=_city_with_solver_ground(mesh,surface)
    np.testing.assert_array_equal(context.vertices[context.faces[context.markers<0]],mesh.vertices[mesh.faces[:1]])
    np.testing.assert_array_equal(context.vertices[context.faces[context.markers>=0]],surface.vertices[surface.faces[1:]])
    np.testing.assert_array_equal(surface.vertices,original)
    assert context.transform.srs=='EPSG:3006'


@pytest.mark.usefixtures('real_worker')
def test_sampling_agrees_with_native_core_terrain_interpolation():
    from dtcc_core import builder
    raster=Raster(data=np.array([[10.,12.,13.],[15.,17.,19.],[19.,20.,23.]]),
                  georef=Affine(1.,0.,0.,0.,-1.,3.))
    mesh=builder.build_terrain_surface_mesh(raster,max_mesh_size=.6,smoothing=0)
    np.testing.assert_allclose(_sample_dem(raster,mesh.vertices[:,:2]),mesh.vertices[:,2],rtol=0,atol=1e-12)


@pytest.mark.usefixtures('real_worker')
@pytest.mark.simulation
def test_dry_ground_pulse_moves_downhill_and_conserves_water(tmp_path):
    from dtcc_core import RectangleMesh
    from dtcc_sim import ShallowWaterParameters, ShallowWaterSolver

    mesh=RectangleMesh(16,16)
    mesh.vertices[:,:2]*=2.
    mesh.vertices[:,2]=10.+.05*mesh.vertices[:,0]
    params=ShallowWaterParameters(bed_from_mesh=True,h0=0.,amplitude=.2,sigma=.2,
        compact_pulse=True,x_center=1.,y_center=1.,final_time=.15,cfl=.16,n_outputs=2)
    result=ShallowWaterSolver(mesh=mesh,params=params,output_dir=tmp_path).run()
    assert result.diagnostics['positive_depth']
    assert result.diagnostics['relative_mass_error']<1e-11
    with np.load(result.case_dir/'solver/state.npz') as raw:
        x=raw['coordinates'][:,0]
        initial=raw['initial_state'][raw['h_dofs']]
        final=raw['state'][raw['h_dofs']]
        cells=raw['cell_dofs']
        # Equal-area triangles make the exact integral of x*h a local mass product.
        def center(h):
            local_x=x[cells];local_h=h[cells]
            integral=(local_x.sum(axis=1)*local_h.sum(axis=1)+(local_x*local_h).sum(axis=1))/12.
            return integral.sum()/local_h.mean(axis=1).sum()
        assert center(final)<center(initial)-1e-4
        newly_wet=(initial[cells].mean(axis=1)==0.)&(final[cells].mean(axis=1)>1e-5)
        assert newly_wet.any()
