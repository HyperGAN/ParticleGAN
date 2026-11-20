"""Summarize deterministic integration sensitivity without changing any grade."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-failure-diagnosis-20261022')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    paths = {name: RAW / filename for name, filename in {
        'order3': 'saved-force-order3.stdout.json', 'order5': 'saved-force-order5.stdout.json',
        'mmd_axes': 'mmd-projection-axes.stdout.json', 'mmd_corners': 'mmd-projection-corners.stdout.json'}.items()}
    data = {name: json.loads(path.read_text()) for name,path in paths.items()}
    for name,d in data.items():
        assert d['training_updates_added'] == d['random_draws_added'] == 0
        assert d['ambient_rng_unchanged'] and not d['qualification_input']
        expected = 'mmd_projection_probe.py' if name.startswith('mmd') else 'saved_force_probe.py'
        assert d['probe_sha256'] == sha(HERE / expected)
    covariance_keys = ('original_covariance_error_drift', 'radial_variance_drift', 'tangent_variance_drift', 'center_squared_error_drift')
    force = {}
    for role in data['order3']['cohorts']:
        row = {}
        for order in ('order3', 'order5'):
            d = data[order]['cohorts'][role]
            assert d['checkpoint_sha256'] == data['order3']['cohorts'][role]['checkpoint_sha256']
            row[order] = {method: {key: d['adversarial_projected_drift'][method]['combined']['mean_over_modes'][key] for key in covariance_keys}
                for method in ('current_frozen_denominator', 'saved_momentum')}
            row[order]['unprojected'] = {key: d['unprojected_adversarial_drift']['mean_over_modes'][key] for key in covariance_keys}
        row['current_covariance_drift_sign_agrees'] = (row['order3']['current_frozen_denominator']['original_covariance_error_drift'] > 0) == (row['order5']['current_frozen_denominator']['original_covariance_error_drift'] > 0)
        force[role] = row
    mmd = {}
    for panel in ('mmd_axes', 'mmd_corners'):
        d = data[panel]
        assert d['core_probe_sha256'] == sha(HERE / 'saved_force_probe.py')
        assert d['analytic_force_max_abs_error'] < 1e-10
        mmd[panel] = {'finite_panel': d['finite_panel'], 'directions': {}}
        for name in ('adversarial', 'mmd', 'combined'):
            mmd[panel]['directions'][name] = {projection: {key: d['directions'][name][projection]['mean_over_modes'][key] for key in covariance_keys}
                for projection in ('unprojected', 'G', 'prior', 'combined')}
        for projection in ('unprojected', 'G', 'prior', 'combined'):
            for key in covariance_keys:
                values = mmd[panel]['directions']
                assert abs(values['combined'][projection][key] - values['adversarial'][projection][key] - values['mmd'][projection][key]) < 1e-9
    result = {'schema_version': 1, 'scope': 'saved_endpoint_sensitivity_summary', 'qualification_input': False,
        'training_updates_added': 0, 'random_draws_added': 0,
        'source_hashes': {name: sha(HERE / name) for name in ('saved_force_probe.py', 'mmd_projection_probe.py', 'summarize_probes.py')},
        'raw_archives': {name: {'path': str(path), 'sha256': sha(path)} for name,path in paths.items()},
        'diagnostic_wall_seconds_by_final_receipt': {name:d['diagnostic_wall_seconds'] for name,d in data.items()},
        'force_quadrature_sensitivity': force, 'mmd_finite_panel_sensitivity': mmd,
        'interpretation_limits': ['Orders3 and5 are integration diagnostics, not separate seed trials or proof of converged quadrature.',
            'Axes/corners finite panels match Gaussian first/second moments but neither is the original random law.',
            'A directional sign predicts only local frozen-model behavior and does not reject or qualify a trained recipe.',
            'Diagnostic CPU wall times are distinct from original paid training charges; earlier debugging executions are not included.']}
    targets = {}
    for name, filename, sensitivity_key in [('force', 'saved-force-diagnosis.json', 'quadrature_sensitivity'),
            ('mmd', 'mmd-projection-diagnosis.json', 'finite_panel_sensitivity')]:
        target = HERE / filename
        current = json.loads(target.read_text())
        section = {k: v for k,v in result.items() if k not in ('force_quadrature_sensitivity', 'mmd_finite_panel_sensitivity')}
        prefix = 'mmd' if name == 'mmd' else 'order'
        section['raw_archives'] = {k:v for k,v in section['raw_archives'].items() if k.startswith(prefix)}
        section['diagnostic_wall_seconds_by_final_receipt'] = {k:v for k,v in section['diagnostic_wall_seconds_by_final_receipt'].items() if k.startswith(prefix)}
        section['comparisons'] = mmd if name == 'mmd' else force
        current[sensitivity_key] = section
        target.write_text(json.dumps(current, indent=2, sort_keys=True, allow_nan=False) + '\n')
        targets[filename] = sha(target)
    print(json.dumps({'outputs': targets, 'status': 'PASS'}, sort_keys=True))


if __name__ == '__main__':
    main()
