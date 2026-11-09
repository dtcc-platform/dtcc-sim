"""Check the independent reference before using it to judge DG evolution."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('wave_reference', Path(__file__).resolve().parents[1]/'tools/verification/shallow_water/wave.py')
wave = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wave)


def test_wave_reference_initial_data_and_reflecting_walls():
    points = np.array([[.2,.3], [.5,.5], [.75,.8]])
    initial = wave.reference(points, 0.)
    np.testing.assert_allclose(initial[:,0], np.exp(-.5*np.sum(((points-.5)/.1)**2,axis=1)), atol=2e-7, rtol=0)
    np.testing.assert_array_equal(initial[:,1:], 0.)
    walls = np.array([[0.,.3],[1.,.7],[.2,0.],[.8,1.]])
    state = wave.reference(walls,.04)
    np.testing.assert_allclose(state[:2,1],0.,atol=1e-14)
    np.testing.assert_allclose(state[2:,2],0.,atol=1e-14)


def test_wave_reference_satisfies_evolution_equations():
    points = np.array([[.2,.3],[.5,.5],[.75,.8]])
    dt = 1e-6
    time_derivative = (wave.reference(points,.04+dt)-wave.reference(points,.04-dt))/(2*dt)
    derivatives = []
    for axis in range(2):
        shift=np.zeros(2); shift[axis]=dt
        derivatives.append((wave.reference(points+shift,.04)-wave.reference(points-shift,.04))/(2*dt))
    c=np.sqrt(9.81)
    np.testing.assert_allclose(time_derivative[:,0]+c*(derivatives[0][:,1]+derivatives[1][:,2]),0.,atol=1e-6)
    np.testing.assert_allclose(time_derivative[:,1]+c*derivatives[0][:,0],0.,atol=1e-6)
    np.testing.assert_allclose(time_derivative[:,2]+c*derivatives[1][:,0],0.,atol=1e-6)
    assert np.linalg.norm(wave.reference(points,.04)-wave.reference(points,0.)) > .1
