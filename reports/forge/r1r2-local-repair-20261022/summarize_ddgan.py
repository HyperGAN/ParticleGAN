"""Compact original DDGAN failure and saved-field receipts; never regrade a run."""
import hashlib
import json
from pathlib import Path

from ddgan_critic_field_probe import INPUT_HASHES, ROOT, SOURCE, sha

HERE = Path(__file__).resolve().parent
ARTIFACTS = Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-ddgan-diagnosis-20261022')
REPO = HERE.parents[2]


def main():
    for relative, digest in INPUT_HASHES.items():
        assert sha(ROOT / relative) == digest
    raw = json.loads((ROOT / 'raw-result.json').read_text())
    receipt = json.loads((ROOT / 'api/receipt.json').read_text())
    records = []
    for order in (2, 3):
        path = ARTIFACTS / f'final-critic-field-order{order}.json'
        record = json.loads(path.read_text())
        assert record['driver_sha256'] == sha(HERE / 'ddgan_critic_field_probe.py')
        assert record['checkpoint_sha256'] == INPUT_HASHES['api/final-state.pt']
        assert record['request_sha256'] == INPUT_HASHES['request.json']
        assert record['raw_result_sha256'] == INPUT_HASHES['raw-result.json']
        assert record['source_digest'] == SOURCE
        records.append(dict(order=order, raw_path=str(path), raw_sha256=sha(path),
            fake_latent_points=record['fake_latent_points'],
            gradient_norms=record['gradient_norms'], wall_seconds=record['wall_seconds'],
            opposed_contexts=sum(p['initial_alignment'] < 0 for p in record['panels']),
            positive_alignment_derivatives={key: sum(p['alignment_derivative'][key] > 0
                for p in record['panels']) for key in record['panels'][0]['alignment_derivative']},
            panels=record['panels']))
    a, b = records
    assert [p['label'] for p in a['panels']] == [p['label'] for p in b['panels']]
    sign = lambda x: (x > 0) - (x < 0)
    same_field = all(sign(p['initial_alignment']) == sign(q['initial_alignment'])
                     for p, q in zip(a['panels'], b['panels']))
    same_directions = {key: all(sign(p['alignment_derivative'][key]) ==
        sign(q['alignment_derivative'][key]) for p, q in zip(a['panels'], b['panels']))
        for key in a['panels'][0]['alignment_derivative']}
    observations = [o for o in receipt['observations'] if o['step'] > 0]
    live = raw['evidence']['live']
    keys = ('projection_cdf', 'normalized_mean', 'covariance_eigen_min', 'covariance_eigen_max')
    panels = ['final'] + [f't{t}_q{q}' for t in range(1, 5) for q in range(2)]
    historical_path = REPO / 'reports/forge/ddgan-failure-investigation/results.json'
    historical = json.loads(historical_path.read_text())
    historical_ddgan = historical['tasks']['model_ddgan_gaussian_chain']
    old_endpoint = historical_ddgan['saved_terminal_suffix']['endpoint_metrics']
    result = dict(schema='r1r2_ddgan_saved_diagnosis_v1', qualification_input=False,
        training_updates_added=0, random_draws_added=0, regraded=False,
        attempt='32a62170283a41158410c13744709fb8', artifact_root=str(ROOT),
        source_digest=SOURCE, original_input_sha256=INPUT_HASHES,
        probe_sha256=sha(HERE / 'ddgan_critic_field_probe.py'),
        summary_source_sha256=sha(__file__),
        original_verdict=receipt['verdict'],
        original_contract=dict(seed=receipt['seed'], completed_updates=receipt['completed_updates'],
            batch_size=receipt['case']['batch_size'], eval_samples=receipt['case']['eval_samples'],
            observations=len(observations), terminal_observations=5,
            recipe=receipt['recipe'], prior=raw['prior_contract'],
            initializer=raw['initializer'], sampling=receipt['case']['sampling'],
            batch_sequence_sha256=raw['evidence']['batch_sequence_sha256']),
        original_endpoint={panel: {key: live[f'{panel}_{key}'] for key in keys} for panel in panels},
        original_trace=dict(passing_joint_observations=sum(o['passed'] for o in observations),
            t1_cdf_pass_counts={f'q{q}': sum(o['metrics'][f't1_q{q}_projection_cdf'] <= .12
                for o in observations) for q in (0, 1)},
            final_cdf_range=[min(o['metrics']['final_projection_cdf'] for o in observations),
                             max(o['metrics']['final_projection_cdf'] for o in observations)],
            terminal_final_cdf=[o['metrics']['final_projection_cdf'] for o in observations[-5:]]),
        historical_comparison=dict(source=historical['source'], report_sha256=sha(historical_path),
            original_publication_sha256=historical['original_publication_sha256'],
            checkpoint_sha256=historical_ddgan['original_artifact_hashes']['final-state.pt'],
            prior_recipe='bispf-optimized-quintic-v1', original_verdict='FAIL',
            final_metrics={key: old_endpoint[f'final_{key}'] for key in keys},
            same_scientific_cohort=False,
            limitations='Different source, runtime and recipe; no matched successful model-native recipe was found.'),
        quadrature=records,
        quadrature_sensitivity=dict(all_field_signs_agree=same_field,
            direction_sign_agreement=same_directions),
        wall_seconds=sum(r['wall_seconds'] for r in records),
        conclusions=[
            'Failure includes both final ancestral sampling and both step-1 conditional panels.',
            'High-noise reverse variance masks clean-predictor variance errors; step 1 has no reverse noise.',
            'The saved NS output field opposes mean correction in seven of eight fixed contexts at both quadrature orders.',
            'Stored critic momentum improves all eight alignments locally, while fresh R1+R2 and R1-only directions worsen t1_q0.',
            'These local controls do not support declaring a D-momentum or R2-removal repair; a full matched global comparison must decide net effects.'],
        limits=record['limits'] + [
            'No new contexts, evaluation samples, qualification gates, continuation or task tuning were introduced.',
            'Orders 2 and 3 are deterministic integration sensitivity, not independent validation.',
            'Small high-noise alignments are numerical point estimates, not significance claims.'])
    (HERE / 'ddgan-diagnosis.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(dict(output='ddgan-diagnosis.json', all_field_signs_agree=same_field,
        direction_sign_agreement=same_directions, training_updates_added=0)))


if __name__ == '__main__':
    main()
