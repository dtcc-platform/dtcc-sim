"""Safety checks for the isolated runtime installer."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('helmholtz_installer', Path(__file__).parents[1] / 'tools/install_helmholtz.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


@pytest.fixture
def vtk_prefix(tmp_path):
    prefix = tmp_path / 'env'
    site = prefix / 'lib/python3.12/site-packages'
    site.mkdir(parents=True)
    egg = site / 'vtk-9.7.0.egg-info'
    egg.write_text('Metadata-Version: 2.1\nName: vtk\nVersion: 9.7.0\n')
    dist = site / 'vtk-9.7.0.dist-info'
    dist.mkdir()
    (dist / 'METADATA').write_text('Metadata-Version: 2.1\nName: vtk\nVersion: 9.7.0\nSummary: VTK is an open-source toolkit for 3D computer graphics, image processing, and visualization\n')
    (dist / 'INSTALLER').write_text('conda\n')
    metadata = prefix / 'conda-meta'
    metadata.mkdir()
    owner = metadata / 'vtk.json'
    owner.write_text(json.dumps({'name': 'vtk-base', 'version': '9.7.0', 'build': 'py312h3887a20_1',
        'files': [str(egg.relative_to(prefix)), str((dist / 'METADATA').relative_to(prefix)), str((dist / 'INSTALLER').relative_to(prefix))]}))
    return prefix, egg, dist, owner


def test_metadata_workaround_preserves_bytes_and_ownership(vtk_prefix, tmp_path):
    prefix, egg, dist, owner = vtk_prefix
    records = {p.name: p.read_bytes() for p in dist.iterdir()}
    owner_before = owner.read_bytes()
    result = installer.normalize_vtk_metadata(prefix, tmp_path / 'backup')
    assert result['applied'] and egg.is_file() and not dist.exists()
    backup = Path(result['preserved_at'])
    assert {p.name: p.read_bytes() for p in backup.iterdir()} == records
    assert owner.read_bytes() == owner_before
    assert installer.normalize_vtk_metadata(prefix, tmp_path / 'second') == {'applied': False}


@pytest.mark.parametrize('change', ['wheel', 'extra_file', 'different_owner', 'different_bytes', 'different_version'])
def test_metadata_workaround_refuses_unrecognized_records(vtk_prefix, tmp_path, change):
    prefix, egg, dist, owner = vtk_prefix
    if change == 'wheel':
        (dist / 'WHEEL').write_text('Wheel-Version: 1.0\n')
    elif change == 'extra_file':
        (dist / 'RECORD').write_text('unexpected')
    elif change == 'different_owner':
        data = json.loads(owner.read_text()); data['build'] = 'another-build'; owner.write_text(json.dumps(data))
    elif change == 'different_bytes':
        with (dist / 'METADATA').open('a') as f: f.write('Requires-Python: >=3.12\n')
    else:
        egg.write_text('Name: vtk\nVersion: 9.8.0\n')
    with pytest.raises(RuntimeError):
        installer.normalize_vtk_metadata(prefix, tmp_path / 'backup')
    assert egg.exists() and dist.exists() and not (tmp_path / 'backup').exists()


def test_lock_refuses_changed_inputs(tmp_path, monkeypatch):
    value = tmp_path / 'input.txt'; value.write_text('expected')
    (tmp_path / 'SHA256SUMS').write_text(hashlib.sha256(value.read_bytes()).hexdigest() + '  input.txt\n')
    monkeypatch.setattr(installer, 'LOCK', tmp_path)
    installer.verify_lock()
    value.write_text('changed')
    with pytest.raises(RuntimeError, match='checksum mismatch'):
        installer.verify_lock()


def test_lock_refuses_path_escape(tmp_path, monkeypatch):
    root = tmp_path / 'lock'; root.mkdir()
    outside = tmp_path / 'outside.txt'; outside.write_text('data')
    (root / 'SHA256SUMS').write_text(hashlib.sha256(outside.read_bytes()).hexdigest() + '  ../outside.txt\n')
    monkeypatch.setattr(installer, 'LOCK', root)
    with pytest.raises(RuntimeError):
        installer.verify_lock()


def test_verification_helper_import_does_not_create_output(tmp_path, monkeypatch):
    destination = tmp_path / 'verification'
    monkeypatch.setattr(__import__('sys'), 'argv', ['verify.py', str(destination)])
    helper_spec = importlib.util.spec_from_file_location('verification_helper_import', Path(__file__).parents[1] / 'tools/verify_helmholtz_install.py')
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    assert not destination.exists()
