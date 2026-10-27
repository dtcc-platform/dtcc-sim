"""Build and verify a pinned Helmholtz runtime in a new Linux x86-64 prefix."""
from pathlib import Path
import argparse
import email.parser
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'requirements/helmholtz-linux-64'
TESTS = ['pou', 'partitioning', 'slice', 'interface', 'manufactured', 'numerical', 'api', 'results', 'launchers', 'mesh_workflow', 'dataset']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def verify_lock():
    for line in (LOCK / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = (LOCK / name).resolve()
        if not path.is_relative_to(LOCK.resolve()) or digest(path) != expected:
            raise RuntimeError(f'Installation input checksum mismatch: {name}')


def normalize_vtk_metadata(prefix, backup):
    """Preserve only the valid legacy record for one known Conda packaging defect."""
    site = Path(prefix) / 'lib/python3.12/site-packages'
    egg = site / 'vtk-9.7.0.egg-info'
    dist = site / 'vtk-9.7.0.dist-info'
    if not egg.exists() or not dist.exists():
        return {'applied': False}
    parse = email.parser.Parser().parsestr
    a, b = parse(egg.read_text()), parse((dist / 'METADATA').read_text())
    if not (a['Name'] == b['Name'] == 'vtk' and a['Version'] == b['Version'] == '9.7.0'):
        raise RuntimeError('Unexpected duplicate VTK metadata; refusing to modify it')
    if (dist / 'WHEEL').exists() or (dist / 'INSTALLER').read_text().strip() != 'conda':
        raise RuntimeError('VTK record is not the known incomplete Conda metadata')
    files = sorted(p.name for p in dist.iterdir())
    if files != ['INSTALLER', 'METADATA']:
        raise RuntimeError('Unexpected files in VTK metadata; refusing to move it')
    owners = []
    for path in (Path(prefix) / 'conda-meta').glob('*.json'):
        item = json.loads(path.read_text())
        owned = item.get('files', [])
        if str(egg.relative_to(prefix)) in owned:
            if str((dist / 'METADATA').relative_to(prefix)) not in owned:
                raise RuntimeError('Duplicate metadata has different package ownership')
            owners.append(item)
    if len(owners) != 1 or any(owners[0].get(k) != v for k, v in {
        'name': 'vtk-base', 'version': '9.7.0', 'build': 'py312h3887a20_1'
    }.items()):
        raise RuntimeError('VTK owner does not match the recorded Conda build')
    expected = {
        'INSTALLER': 'bc33022edcb7639ff53355b4e91dade50a0bbf0299efeb6171d1ec0ba5029cfc',
        'METADATA': '7a9b360382de8da9d9cf4639398825a93daa31d33aea61e45b19c60cde8608af',
    }
    if {p.name: digest(p) for p in dist.iterdir()} != expected:
        raise RuntimeError('VTK metadata bytes differ from the recorded defect')
    backup = Path(backup)
    backup.mkdir()
    destination = backup / dist.name
    dist.rename(destination)
    return {'applied': True, 'preserved_at': str(destination), 'sha256': expected,
            'legacy_record_retained': str(egg), 'package_binaries_modified': False,
            'conda_ownership_database_modified': False,
            'reversal': 'Move the preserved dist-info back to its original site-packages path; this restores the original duplicate-metadata condition.'}


class Installation:
    def __init__(self, args):
        self.args = args
        self.work = Path(args.work_dir).resolve()
        self.prefix = self.work / 'env'
        self.python = self.prefix / 'bin/python'
        self.uv = self.prefix / 'bin/uv'
        self.env = dict(os.environ)
        for key in list(self.env):
            if key.startswith(('HELMHOLTZ_', 'CONDA_')) or key in (
                'PYTHONPATH', 'PYTHONHOME', 'PETSC_DIR', 'SLEPC_DIR', 'LD_LIBRARY_PATH'
            ):
                self.env.pop(key, None)
        self.env.update(PYTHONNOUSERSITE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                        MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', CMAKE_BUILD_PARALLEL_LEVEL='2')
        self.conda = str(Path(args.conda).resolve())

    def status(self, state, phase, **extra):
        write(self.work / 'status.json', dict(state=state, phase=phase, **extra))

    def run(self, phase, argv, cwd=None, activate=False):
        self.status('running', phase)
        argv = list(map(str, argv))
        if activate:
            argv = [self.conda, 'run', '--no-capture-output', '-p', str(self.prefix), *argv]
        write(self.work / f'{phase}-command.json', {'argv': argv, 'cwd': str(cwd or self.work)})
        with (self.work / f'{phase}.log').open('x') as log:
            process = subprocess.Popen(argv, cwd=cwd or self.work, env=self.env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            write(self.work / f'{phase}-process.json', {'pid': process.pid})
            code = process.wait()
        (self.work / f'{phase}-exit-code.txt').write_text(str(code) + '\n')
        if code:
            raise RuntimeError(f'{phase} failed with exit {code}; inspect {phase}.log')

    def install(self):
        start = time.monotonic()
        try:
            verify_lock()
            write(self.work / 'inputs.json', {
                'lock_sha256': digest(LOCK / 'SHA256SUMS'),
                'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in ROOT.rglob('*.py')
                    if not any(x in p.parts for x in ('.git', '.agent', '__pycache__', '.venv', 'build', 'dist'))},
                'platform': platform.platform(), 'python_bootstrap': sys.version})
            self.run('create-conda', [self.conda, 'create', '--yes', '--copy', '--no-default-packages',
                     '--prefix', self.prefix, '--file', LOCK / 'conda-explicit.txt'])
            for name, requirements in [('install-python', 'python.txt'), ('install-build-tools', 'build.txt')]:
                self.run(name, [self.uv, 'pip', 'install', '--python', self.python, '--no-deps',
                    '--index-url', 'https://pypi.org/simple', '-r', LOCK / requirements], activate=True)
            metadata = normalize_vtk_metadata(self.prefix, self.work / 'metadata-backup')
            write(self.work / 'metadata-fix.json', metadata)
            sources = json.loads((LOCK / 'sources.json').read_text())
            src = self.work / 'src'
            src.mkdir()
            for name, pin in sources.items():
                self.run('clone-' + name, ['git', 'clone', '--filter=blob:none', '--no-checkout', pin['url'], src / name])
                self.run('checkout-' + name, ['git', 'checkout', '--detach', pin['commit']], cwd=src / name)
            shutil.copytree(ROOT, src / 'sim', symlinks=True,
                ignore=shutil.ignore_patterns('.git', '.agent', '.pytest_cache', '__pycache__', '.venv', 'build', 'dist', '*.egg-info'))
            wheels = self.work / 'wheelhouse'
            wheels.mkdir()
            for name in ['mesher', 'tetgen', 'core', 'sim']:
                self.run('build-' + name, [self.python, '-m', 'pip', 'wheel', '--no-deps',
                    '--no-build-isolation', '--wheel-dir', wheels, src / name], activate=True)
            self.run('install-projects', [self.uv, 'pip', 'install', '--python', self.python,
                      '--no-deps', *sorted(wheels.glob('*.whl'))], activate=True)
            self.run('dependency-check', [self.uv, 'pip', 'check', '--python', self.python], activate=True)
            self.run('readiness', [self.python, '-m', 'dtcc_sim.helmholtz.check_runtime', '--workers', '2', '--mesh-generation'], activate=True)
            if self.args.verify:
                tests = [f'tests/test_helmholtz_{name}.py' for name in TESTS]
                tests.append('tests/test_helmholtz_installer.py')
                self.run('focused-tests', [self.python, '-m', 'pytest', '-q', *tests], cwd=src / 'sim', activate=True)
                self.run('real-workflow', [self.python, '-u', src / 'sim/tools/verify_helmholtz_install.py', self.work / 'verification'], activate=True)
                assert json.loads((self.work / 'verification/smoke-validation.json').read_text())['passed']
            self.run('package-inventory', [self.conda, 'list', '-p', self.prefix, '--json'])
            write(self.work / 'comparison.json', {'passed': True, 'prefix': str(self.prefix),
                'verification_requested': self.args.verify, 'metadata_workaround': metadata,
                'wheel_sha256': {p.name: digest(p) for p in wheels.glob('*.whl')},
                'focused_tests': (int(re.search(r'(\d+) passed', (self.work / 'focused-tests.log').read_text()).group(1)) if self.args.verify else None),
                'seconds': time.monotonic() - start,
                'scope': 'Pinned new Linux x86-64 prefix and native builds from official source revisions. Downloads/geodata may use caches. No historical partition or performance claim.'})
            self.status('completed', 'complete', passed=True)
            code = 0
        except BaseException:
            error = traceback.format_exc()
            (self.work / 'error.txt').write_text(error)
            self.status('failed', 'exception', error=error)
            code = 1
        (self.work / 'exit-code.txt').write_text(str(code) + '\n')
        return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True, help='New output directory; existing paths are refused')
    parser.add_argument('--conda', default=shutil.which('conda'), help='Conda executable')
    parser.add_argument('--verify', action='store_true', help='Run focused tests and small real mesh/solve/export checks')
    parser.add_argument('--detach', action='store_true', help='Continue after terminal or SSH disconnect')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        parser.error('This recorded recipe supports Linux x86-64 only')
    if not args.conda or not Path(args.conda).is_file():
        parser.error('Provide an existing Conda executable with --conda')
    work = Path(args.work_dir).resolve()
    if work == ROOT or ROOT.is_relative_to(work) or work.is_relative_to(ROOT):
        parser.error('Place --work-dir outside the source checkout')
    if not args.worker:
        work.mkdir(parents=True, exist_ok=False)
    if args.detach and not args.worker:
        argv = [sys.executable, str(Path(__file__).resolve()), '--work-dir', str(work), '--conda', args.conda, '--worker']
        if args.verify:
            argv.append('--verify')
        with (work / 'driver.log').open('x') as log:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(work / 'launch.json', {'pid': process.pid, 'session_id': process.pid})
        print(json.dumps({'directory': str(work), 'pid': process.pid}))
        return 0
    return Installation(args).install()


if __name__ == '__main__':
    raise SystemExit(main())
