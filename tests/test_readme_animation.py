"""Keep the public README illustration aligned with the selected default and its evidence."""
import hashlib
import json
from pathlib import Path

from particlegan import get_recipe

ROOT = Path(__file__).resolve().parents[1]


def test_gaussian_animation_preserves_the_current_default_recipe():
    receipt = json.loads((ROOT / 'reports/readme-100gaussians/summary-current.json').read_text())
    default = get_recipe()
    prior = receipt['prior']
    expected = json.loads(json.dumps(default.replace(
        prior_kind=prior['kind'] if prior['kind'] == 'mog' else 'particles',
        sigma_rel=0.0,
        standardize=prior['standardize'],
        num_particles=prior['num_particles'],
    ).to_dict()))
    assert receipt['default_recipe'] == default.name == 'unified_v11'
    assert receipt['recipe'] == expected, 'Current README illustration must match get_recipe() + declared task prior'
    assert prior == {
        'kind': 'mog',
        'class_name': 'MoGParticlePrior',
        'num_particles': 20000,
        'z_dim': 2,
        'sigma': 0.025,
        'standardize': False,
        'learnable': True,
    }
    assert receipt['completed_steps'] == receipt.get('execution_budget', expected['total_steps'])
    assert receipt['completed_steps'] >= expected['total_steps']
    assert receipt['seed'] == 0
    assert receipt['task']['initialization'] == {'generator': 0, 'critic': 1, 'prior': 2}
    archived = receipt['artifact_archive']['archived_090_20k_particle_cloud']
    assert archived['commit'] == '1bc01a7bf4b151c01073c35f59a1d894d0917571'
    assert archived['gif_blob'] == '4f97aee33969d391182338dd41a90e3c896eb11a'
    assert archived['summary_blob'] == 'f6f31cc543f7960c6c79d95d3e2a7e7db4eecb5b'


def test_readme_current_default_illustration_uses_actual_candidate_training():
    base = ROOT / 'reports/forge/bispf-throughput'
    path = 'media/bispf-throughput-v1/img_stripes2.gif'
    receipt = json.loads((base / path).with_suffix('.json').read_text())
    readout = json.loads((base / 'results.json').read_text())
    row = next(item for item in readout['attempts'] if item.get('media', {}).get('path') == path)
    assert row['candidate'] == readout['candidate'] == 'bispf-throughput-v1'
    assert row['status'] == receipt['recorded_grade'] == 'PASS'
    assert row['cost']['completed_steps'] == 600
    assert receipt['kind'] == 'actual_training_saved_observations_gif'
    assert receipt['optimizer_updates_added'] == receipt['sampling_draws_added'] == 0
    assert hashlib.sha256((base / path).read_bytes()).hexdigest() == receipt['gif_sha256']
    assert str((base / path).relative_to(ROOT)) in (ROOT / 'README.md').read_text()
    assert get_recipe('bispf_optimized').optimizer_svd_backend == 'batched'
    assert get_recipe('bispf_optimized').training_precision == 'bf16_channels_last'


def test_readme_animation_preserves_its_recorded_verdict_and_gif():
    receipt = json.loads((ROOT / 'reports/readme-100gaussians/summary-current.json').read_text())
    animation = ROOT / '100gaussians.gif'
    assert hashlib.sha256(animation.read_bytes()).hexdigest() == receipt['render']['gif_sha256']
    assert receipt['render']['observations_sha256'] == receipt['observations_sha256']
    assert receipt['render']['frame_steps'] == receipt['media_steps']
    assert receipt['media_steps'][0] == 0
    assert receipt['media_steps'][-1] == receipt['completed_steps']
    assert len(receipt['terminal_checks']) == receipt['gate']['terminal_checks']
    passed = all(row['modes'] == 100 and row['hq'] >= .95 for row in receipt['terminal_checks'])
    assert receipt['status'] == ('PASS' if passed else 'FAIL')
    assert receipt['render']['status'] == receipt['status']
