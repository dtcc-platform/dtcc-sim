"""Summarize timing and field comparisons from flow element benchmarks."""

import argparse
import json
from pathlib import Path
import statistics

import numpy as np


def summarize(root):
    rows = []
    for path in sorted(root.glob('*/measurements.json')):
        data = json.loads(path.read_text())
        records = data['records']
        warm = records[1:] if len(records) >= 4 else records
        row = {k:data[k] for k in ['case','element','condensed','repaired','cells','mesh_sha256']}
        gate = lambda r: r['accepted'] and r['diagnostics']['convergence']['dimensionless_divergence'] <= (.005 if data['case'] == 'channel' else .1) and all(v < .002 for v in r.get('errors', {}).values()) and r['diagnostics']['convergence']['wall_normal_velocity_rms'] <= 1e-8
        row.update(label=path.parent.name, warm_runs=len(warm) if len(records)>=4 else 0, samples=len(warm), aggregation='warm_median' if len(records)>=4 else 'single_or_partial', accepted=all(gate(r) for r in warm))
        keys = ['cpu_s','peak_rss_bytes','wall_s','jit_s','system_creation_s','configuration_s','residual_assembly_s','jacobian_assembly_s','linear_total_s','condensation_s','condensed_factor_setup_s','condensed_rhs_s','condensed_triangular_s','bubble_recovery_s','full_dofs','retained_dofs']
        row.update({key:statistics.median(r[key] for r in warm) for key in keys if all(key in r for r in warm)})
        for key in ['MatLUFactorSym','MatLUFactorNum','MatSolve']:
            row[key] = statistics.median(r['petsc_events'][key]['time'] for r in warm)
        if all(r.get('jacobian_count') for r in warm):
            row['jacobian_per_assembly_s'] = statistics.median(r['jacobian_assembly_s']/r['jacobian_count'] for r in warm)
        if all('nonlinear_iterations' in r.get('diagnostics',{}) for r in warm):
            row['newton_iterations'] = [r['diagnostics']['nonlinear_iterations'] for r in warm]
        if all('convergence' in r.get('diagnostics',{}) for r in warm):
            row['quality'] = warm[-1]['diagnostics']['convergence']
        if all('errors' in r for r in warm):
            row['errors'] = warm[-1]['errors']
        row['range_wall_s'] = [min(r['wall_s'] for r in warm),max(r['wall_s'] for r in warm)]
        rows.append(row)
    comparisons = []
    reference = root/'city-p2-repaired/probes-1.npz'
    if reference.exists():
        baseline = np.load(reference)
        for path in sorted(root.glob('city-*/probes-*.npz')):
            if path.name not in ['probes-1.npz','probes-0.npz']:
                continue
            if path.name == 'probes-0.npz' and (path.parent/'probes-1.npz').exists():
                continue
            case = np.load(path)
            valid = np.isfinite(baseline['p']) & np.isfinite(case['p'])
            row = dict(label=path.parent.name, common_probes=int(sum(valid)))
            for key in ['u','p']:
                a,b=baseline[key][valid],case[key][valid]
                row[key+'_relative_l2']=float(np.linalg.norm(a-b)/np.linalg.norm(a))
            speed0=np.linalg.norm(baseline['u'][valid],axis=1)
            speed=np.linalg.norm(case['u'][valid],axis=1)
            row['speed_relative_l2']=float(np.linalg.norm(speed-speed0)/np.linalg.norm(speed0))
            row['speed_absolute_rms']=float(np.sqrt(np.mean((speed-speed0)**2)))
            comparisons.append(row)
    equivalence = []
    for path in sorted(root.glob('*-condensed*/fields-0.npz')):
        full_path = root/path.parent.name.replace('-condensed','')/'fields-0.npz'
        if not full_path.exists():
            continue
        a,b=np.load(full_path),np.load(path)
        if any(a[key].shape != b[key].shape for key in ['u','p']):
            continue
        equivalence.append(dict(label=path.parent.name, **{key+'_relative_l2':float(np.linalg.norm(a[key]-b[key])/max(np.linalg.norm(a[key]),1e-30)) for key in ['u','p']}))
    result=dict(timings=rows,city_comparisons=comparisons,condensation_equivalence=equivalence)
    (root/'summary.json').write_text(json.dumps(result,indent=2))
    print('label | accepted | warm runs | total | J assembly | numeric factor | triangular | condensation')
    for row in rows:
        print(' | '.join(str(row.get(key,'')) if key not in ['wall_s','jacobian_assembly_s','MatLUFactorNum','MatSolve','condensation_s'] else f'{row.get(key,0):.4f}' for key in ['label','accepted','warm_runs','wall_s','jacobian_assembly_s','MatLUFactorNum','MatSolve','condensation_s']))
    print(json.dumps(comparisons,indent=2))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    summarize(parser.parse_args().root)
