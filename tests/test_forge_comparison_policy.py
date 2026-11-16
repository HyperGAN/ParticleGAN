"""Historical comparison policy changes counts, never original task verdicts."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from test_forge_current_technique_inventory import ROOT, evidence, publication  # noqa: F401


def test_actual_registered_dispositions_are_a_valid_comparison_projection():
    actual = read_json(ROOT / 'configs/forge/view-history/discriminator_stability-v12.json')
    recorded = read_json(ROOT / 'configs/forge/view-history/discriminator_stability-v9.json')
    publication._validate_comparison_policy(actual, recorded)


@pytest.mark.parametrize('change', ['task', 'tier', 'order', 'eligibility', 'calibration', 'same_revision'])
def test_comparison_policy_refuses_changes_beyond_dispositions(change):
    actual = read_json(ROOT / 'configs/forge/view-history/discriminator_stability-v12.json')
    recorded = read_json(ROOT / 'configs/forge/view-history/discriminator_stability-v9.json')
    if change == 'task':
        recorded['assignments'][0]['task'] = 'another_task'
    elif change == 'tier':
        recorded['assignments'][0]['qualification_tier'] = 2
    elif change == 'order':
        recorded['assignments'].reverse()
    elif change in {'eligibility', 'calibration'}:
        recorded[change] = {'changed': True}
    else:
        recorded['revision'] = actual['revision']
    with pytest.raises(ValueError, match='comparison policy'):
        publication._validate_comparison_policy(actual, recorded)


def comparison_fixture():
    actual = dict(view='stability', view_revision=11, policy_fingerprint='actual',
                  tier_requirements={'1': ['acquire'], '2': [], '3': ['hold']})
    row = dict(candidate_id='optimized', candidate_revision='revision', cohort='cohort',
               attempt_ids=['attempt'], bindings={'trainer_family': 'family', 'task_contracts': {'acquire': 'a', 'held': 'b', 'hold': 'c'}},
               runtime_cohort={'backend': 'cuda'}, cost={'wall_seconds': 10}, status='PASS', qualified_tier=3,
               tiers={'1': {'passed': 1, 'total': 1}, '2': {'passed': 0, 'total': 0}, '3': {'passed': 1, 'total': 1}},
               tasks=[{'task_id': 'acquire', 'status': 'PASS'}, {'task_id': 'hold', 'status': 'PASS'}],
               nonrequired_tasks=[{'task_id': 'held', 'status': 'FAIL'}])
    actual['rows'] = [row]
    projected = deepcopy(actual)
    projected.update(view_revision=9, policy_fingerprint='comparison')
    projected['rows'][0].update(status='FAIL', qualified_tier=1)
    projected['rows'][0]['tasks'].append(projected['rows'][0]['nonrequired_tasks'].pop())
    projected['rows'][0]['tiers']['2'] = {'passed': 0, 'total': 1}
    comparison = dict(policy={'revision': 9}, path='policy.json', sha256='policy-hash',
                      reference_task_contracts={'family': deepcopy(row['bindings']['task_contracts'])})
    return actual, projected, comparison


def test_projection_preserves_original_qualification_and_all_numerical_verdicts():
    actual, projected, comparison = comparison_fixture()
    before = deepcopy(actual)
    publication._comparison_projection(actual, projected, comparison)
    assert actual == before
    assert projected['rows'][0]['qualified_tier'] == 1
    assert projected['rows'][0]['executed_policy_qualification']['qualified_tier'] == 3
    proof = projected['comparison_policy_projection']
    assert proof['actual_qualification']['rows'] == actual['rows']
    assert proof['actual_qualification_sha256'] == stable_hash(proof['actual_qualification'])
    assert proof['qualification_reuse'] is proof['qualification_input'] is False


@pytest.mark.parametrize('change', ['verdict', 'task_contract', 'runtime', 'attempt', 'missing'])
def test_projection_rejects_changed_evidence_or_missing_original(change):
    actual, projected, comparison = comparison_fixture()
    row = projected['rows'][0]
    if change == 'verdict':
        row['tasks'][-1]['status'] = 'PASS'
    elif change == 'task_contract':
        row['bindings']['task_contracts']['acquire'] = 'different-gate-budget-or-sampling'
    elif change == 'runtime':
        row['runtime_cohort'] = {'backend': 'cpu'}
    elif change == 'attempt':
        row['attempt_ids'] = ['foreign']
    else:
        projected['rows'] = []
    with pytest.raises(ValueError, match='comparison'):
        publication._comparison_projection(actual, projected, comparison)


def test_explicit_cached_comparison_preserves_registered_rows_without_regrading(evidence):  # noqa: F811
    root, manifest = evidence
    publication.publish_current(root)
    original = read_json(root / 'reports/forge/technique-inventory.json')['rows']
    manifest_hash = file_hash(root / publication.EVIDENCE_MANIFEST)
    policy_path = root / 'configs/forge/views/discriminator_stability.json'
    recorded_path = root / 'configs/forge/view-history/discriminator_stability-v2.json'
    atomic_json(recorded_path, read_json(policy_path))
    actual = read_json(policy_path)
    actual['revision'] = 3
    atomic_json(policy_path, actual)
    result = publication.publish_current(root, comparison_policy=recorded_path)
    assert read_json(result['json'])['rows'] == original
    assert file_hash(root / publication.EVIDENCE_MANIFEST) == manifest_hash


def test_independent_projection_requires_a_frozen_source(tmp_path):
    with pytest.raises(ValueError, match='exact frozen source'):
        publication.regenerate(tmp_path, comparison_policy=Path('policy.json'))


def test_pending_snapshot_overlay_is_read_only_hash_checked_and_restored_on_failure(tmp_path):
    from experiments.forge.test_performance import Inputs
    pending = tmp_path / 'reports/forge/technique-evidence/new.json'
    text = '{"new_source": true}\n'
    import hashlib
    expected = hashlib.sha256(text.encode()).hexdigest()
    original = Inputs.__init__
    with pytest.raises(RuntimeError, match='render failed'):
        with publication._pending_publication_inputs(tmp_path, (pending, text)):
            inputs = Inputs(tmp_path)
            assert inputs.read('reports/forge/technique-evidence/new.json', sha256=expected) == {'new_source': True}
            with pytest.raises(ValueError, match='bytes changed'):
                inputs.read('reports/forge/technique-evidence/new.json', sha256='wrong')
            assert not pending.exists()
            raise RuntimeError('render failed')
    assert Inputs.__init__ is original
    assert not pending.exists()
    with pytest.raises(FileNotFoundError):
        Inputs(tmp_path).read('reports/forge/technique-evidence/new.json')
    pending.parent.mkdir(parents=True)
    pending.write_text('{}\n')
    with pytest.raises(ValueError, match='conflicts'):
        with publication._pending_publication_inputs(tmp_path, (pending, text)):
            pytest.fail('conflicting immutable source must not be overlaid')
    assert pending.read_text() == '{}\n'


@pytest.mark.parametrize('captured', [False, True])
def test_frozen_catalog_overlay_preserves_captured_presence_and_absence(tmp_path, captured):
    name = 'reports/transfer_suite/unadjusted/leading_profile.json'
    live, frozen = tmp_path / 'live', tmp_path / 'frozen'
    source, target = live / name, frozen / name
    source.parent.mkdir(parents=True)
    source.write_text('{"live": "later optional profile"}\n')
    sibling = source.with_name('readout.json')
    sibling.write_text('{"knowledge": true}\n')
    files = {}
    if captured:
        target.parent.mkdir(parents=True)
        target.write_text('{"frozen": "original profile"}\n')
        files[name] = file_hash(target)
    closure = publication._catalog_support_closure([name], {'recorded-source': {'files': files}})
    publication._overlay_reports(live / 'reports', frozen / 'reports',
                                 excluded=[Path(p).relative_to('reports') for p in closure['preserved_absent_paths']])
    assert target.exists() == captured
    if captured:
        assert target.read_text() == '{"frozen": "original profile"}\n'
        assert closure['captured_present_paths'] == [name]
        assert not target.is_symlink()
    else:
        assert closure['preserved_absent_paths'] == [name]
    assert (target.parent / sibling.name).read_bytes() == sibling.read_bytes()
    assert source.read_text() == '{"live": "later optional profile"}\n'


def test_catalog_closure_retains_original_study_support_and_rejects_conflicting_source_presence():
    name = 'reports/profile.json'
    sources = {'sparse': {'files': {}}, 'full': {'files': {name: 'original-hash'}}}
    with pytest.raises(ValueError, match='conflicting captured'):
        publication._catalog_support_closure([name], sources)
    closure = publication._catalog_support_closure([name], {'sparse': sources['sparse']}, {name: 'study-hash'})
    assert closure['preserved_absent_paths'] == []
    with pytest.raises(ValueError, match='unsafe'):
        publication._catalog_support_closure(['reports/../escape.json'], {'source': {'files': {}}})


def test_catalog_inventory_uses_frozen_profile_rules_in_a_fresh_process(tmp_path):
    package = tmp_path / 'experiments/forge'
    package.mkdir(parents=True)
    (package.parent / '__init__.py').write_text('')
    (package / '__init__.py').write_text('')
    (package / 'views.py').write_text(
        "def load_tasks(root):\n    return {'task': {'evaluation': {'sources': {'reports/evaluator.py': 'hash'}}}}\n")
    (package / 'hostprofiles.py').write_text(
        "def profile_source_paths(task):\n    return ['reports/frozen-profile.json']\n")
    assert publication._frozen_catalog_support(tmp_path) == ['reports/evaluator.py', 'reports/frozen-profile.json']


@pytest.mark.parametrize('missing', ['proof', 'measured', 'both'])
def test_projection_refuses_to_turn_recorded_study_into_silent_unresolved_rows(missing):
    request = {'candidate': {'id': 'optimized'}, 'view': {'id': 'stability'}, 'execution_backend': 'cuda'}
    studies = [{'request': request}]
    result = {'registered_study_reconstruction': [{'candidate_id': 'optimized'}],
              'rows': [{'candidate_id': 'optimized', 'attempt_ids': ['original']} ]}
    publication._validate_recorded_study_projection(studies, result, 'stability', 'cuda')
    if missing in {'proof', 'both'}:
        result['registered_study_reconstruction'] = []
    if missing in {'measured', 'both'}:
        result['rows'] = [{'candidate_id': 'optimized', 'attempt_ids': []}]
    with pytest.raises(ValueError, match='lost recorded study admission or measured rows'):
        publication._validate_recorded_study_projection(studies, result, 'stability', 'cuda')
    publication._validate_recorded_study_projection(studies, result, 'another-view', 'cuda')


@pytest.mark.parametrize('scope', ['research_diagnostic', 'calibration_diagnostic'])
def test_diagnostic_study_does_not_require_an_ordinary_qualifying_row(scope):
    studies = [{'request': {'candidate': {'id': 'diagnostic'}, 'view': {'id': 'view', 'evidence_scope': scope},
                            'execution_backend': 'cuda'}}]
    publication._validate_recorded_study_projection(studies, {'rows': []}, 'view', 'cuda')


def test_additive_view_revision_is_valid_comparison_projection_and_retains_added_tasks():
    actual_view = read_json(ROOT / 'configs/forge/views/discriminator_stability.json')
    recorded_view = read_json(ROOT / 'configs/forge/view-history/discriminator_stability-v9.json')
    publication._validate_comparison_policy(actual_view, recorded_view)

    actual, projected, comparison = comparison_fixture()
    actual['status_reasons'] = {'added-reason': ['new task failure reason']}
    actual['rows'][0]['bindings']['task_contracts']['added_tier1'] = 'added-contract'
    actual['rows'][0]['tasks'].append({'task_id': 'added_tier1', 'status': 'PASS', 'reasons_sha256': 'added-reason'})
    projected['rows'][0]['bindings']['task_contracts']['added_tier1'] = 'added-contract'
    comparison['added_actual_tasks'] = ['added_tier1']
    comparison['verified_evaluator_lineage'] = {'family': {'acquire': 'a-source-refresh'}}
    actual['rows'][0]['bindings']['task_contracts']['acquire'] = 'a-source-refresh'
    projected['rows'][0]['bindings']['task_contracts']['acquire'] = 'a-source-refresh'

    publication._comparison_projection(actual, projected, comparison)
    carried = [t for t in projected['rows'][0]['nonrequired_tasks'] if t['task_id'] == 'added_tier1']
    assert carried == [{'task_id': 'added_tier1', 'status': 'PASS', 'reasons_sha256': 'added-reason'}]
    assert projected['status_reasons']['added-reason'] == ['new task failure reason']
    assert projected['rows'][0]['qualified_tier'] == 1

    # Unauthorized extra task contract without added_actual_tasks must fail closed.
    bad_actual, bad_projected, bad_comparison = comparison_fixture()
    bad_actual['rows'][0]['bindings']['task_contracts']['sneaky'] = 'x'
    bad_projected['rows'][0]['bindings']['task_contracts']['sneaky'] = 'x'
    with pytest.raises(ValueError, match='comparison projection changes recorded task contracts'):
        publication._comparison_projection(bad_actual, bad_projected, bad_comparison)


def test_evaluator_revision_lineage_accepts_source_only_refresh_and_rejects_gate_change():
    from experiments.forge.views import load_tasks, task_evaluation_fingerprint
    recorded = read_json(ROOT / 'reports/forge/technique-inventory.json')
    live_two_pole = load_tasks(ROOT)['two_pole']
    live_eval = task_evaluation_fingerprint(live_two_pole)
    k3p_ref = next(row['bindings']['task_contracts'] for row in recorded['rows'] if row['trainer_family'] == 'k3p')
    ref_eval = recorded['task_contracts'][k3p_ref['two_pole']]['evaluation_sha256']
    assert ref_eval != live_eval
    assert publication._evaluator_revision_reaches(ROOT, live_two_pole, ref_eval) is True

    tampered = deepcopy(live_two_pole)
    tampered['execution']['steps'] += 1
    assert publication._evaluator_revision_reaches(ROOT, tampered, ref_eval) is False

    tampered_gate = deepcopy(live_two_pole)
    tampered_gate['evaluation']['thresholds'][0][2] = 0.99
    assert publication._evaluator_revision_reaches(ROOT, tampered_gate, ref_eval) is False



@pytest.mark.parametrize('damage', [
    'budget', 'ownership', 'sampling', 'gate', 'earlier_revision',
    'parent_sha', 'parent_blob', 'source_pin', 'qualification_flag', 'receipt_schema',
])
def test_combined_source_lineage_rejects_semantic_damage_even_with_rehashed_receipt(tmp_path, monkeypatch, damage):
    import hashlib
    import json

    relative = 'configs/forge/tasks/two_pole.json'
    task = read_json(ROOT / relative)
    binding = task['evaluation']['integration_source_binding']
    assert binding['id'] == 'reviewed-model-prs-integration-20261109'
    receipt_path = binding['integration_receipt']
    receipt = read_json(ROOT / receipt_path)
    row = next(row for row in receipt['tasks'] if row['previous_task_path'] == relative)
    for source in task['evaluation']['sources']:
        path = tmp_path / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(ROOT / source)
    task_path = tmp_path / relative
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_bytes((ROOT / relative).read_bytes())
    atomic_json(tmp_path / receipt_path, receipt)

    # Keep archived Git reads exact while isolating the mutable current task
    # and receipt. No repository file or historical receipt is rewritten.
    original_check_output = publication.subprocess.check_output
    def archived_git(args, **kwargs):
        if args[:2] == ['git', 'show']:
            kwargs['cwd'] = ROOT
        return original_check_output(args, **kwargs)
    monkeypatch.setattr(publication.subprocess, 'check_output', archived_git)
    recorded = read_json(ROOT / 'reports/forge/technique-inventory.json')
    reference = next(row['bindings']['task_contracts'] for row in recorded['rows']
                     if row['trainer_family'] == 'k3p')
    target = recorded['task_contracts'][reference['two_pole']]['evaluation_sha256']
    assert publication._evaluator_revision_reaches(tmp_path, task, target)

    if damage == 'budget':
        task['execution']['steps'] += 1
    elif damage == 'ownership':
        task['execution']['prior_contract'] = 'task_owned'
    elif damage == 'sampling':
        task['evaluation']['sampling_law'] = 'different_sampler'
    elif damage == 'gate':
        task['evaluation']['thresholds'][0][2] = 0.99
    elif damage == 'earlier_revision':
        task['evaluation']['evaluator_revision']['id'] = 'unrelated_ancestry'
    elif damage in {'parent_sha', 'parent_blob'}:
        key = 'previous_task_sha256' if damage == 'parent_sha' else 'previous_task_blob'
        binding[key] = row[key] = '0' * len(binding[key])
    elif damage == 'source_pin':
        source = next(iter(task['evaluation']['sources']))
        task['evaluation']['sources'][source] = row['sources_after'][source] = '0' * 64
    elif damage == 'qualification_flag':
        receipt['qualification_input'] = True
    else:
        receipt['schema'] = 'unverified_source_binding'
    raw = (json.dumps(task, indent=2) + '\n').encode()
    task_path.write_bytes(raw)
    row['current_task_sha256'] = hashlib.sha256(raw).hexdigest()
    atomic_json(tmp_path / receipt_path, receipt)
    assert publication._evaluator_revision_reaches(tmp_path, task, target) is False


def test_candidate_roster_requires_frozen_source_and_cannot_combine_with_declaration_registration(tmp_path):
    with pytest.raises(ValueError, match='declaration roster requires exact frozen source reconstruction'):
        publication.regenerate(tmp_path, candidate_roster=['atlas-particle-normal-v4'])
    with pytest.raises(ValueError, match='cannot combine --declaration-registration and --candidate'):
        publication.regenerate(
            tmp_path,
            source_commit='HEAD',
            declaration_registration=Path('reg.json'),
            candidate_roster=['atlas-particle-normal-v4'],
        )
    with pytest.raises(ValueError, match='declaration roster requires exact frozen source reconstruction'):
        publication.publish_current(tmp_path, candidate_roster=['atlas-particle-normal-v4'])


def test_comparison_contracts_match_is_idempotent_after_additive_row_is_published():
    actual, projected, comparison = comparison_fixture()
    comparison['policy']['assignments'] = [
        {'task': 'acquire', 'qualification_tier': 1},
        {'task': 'held', 'qualification_tier': 2},
        {'task': 'hold', 'qualification_tier': 3},
    ]
    comparison['added_actual_tasks'] = ['added_tier1']
    # Simulate re-running comparison projection once reference_task_contracts
    # already comes from a published later-revision row containing added_tier1.
    comparison['reference_task_contracts']['family']['added_tier1'] = 'added-contract'
    actual['rows'][0]['bindings']['task_contracts']['added_tier1'] = 'added-contract'
    actual['rows'][0]['tasks'].append({'task_id': 'added_tier1', 'status': 'PASS'})
    projected['rows'][0]['bindings']['task_contracts']['added_tier1'] = 'added-contract'

    assert publication._comparison_contracts_match(projected['rows'][0], comparison) is True
    publication._comparison_projection(actual, projected, comparison)
    assert any(t['task_id'] == 'added_tier1' and t['status'] == 'PASS'
               for t in projected['rows'][0]['nonrequired_tasks'])

    # Divergent added-task contract digest must still fail closed.
    bad_row = deepcopy(projected['rows'][0])
    bad_row['bindings']['task_contracts']['added_tier1'] = 'tampered-added-contract'
    assert publication._comparison_contracts_match(bad_row, comparison) is False

    # Missing a required comparison-policy task in reference must also fail closed.
    incomplete_comparison = deepcopy(comparison)
    incomplete_comparison['reference_task_contracts']['family'].pop('acquire')
    assert publication._comparison_contracts_match(projected['rows'][0], incomplete_comparison) is False


def test_measured_candidates_at_source_discovers_non_study_ordinary_attempts(tmp_path):
    manifests = {'src-digest': {'origin_commit': 'a' * 40, 'digest': 'src-digest', 'files': {}}}
    attempts = tmp_path / 'reports/forge/attempts'
    for name, request in [
        ('01_ordinary_non_study', {
            'candidate': {'id': 'non-study-cand'},
            'view': {'id': 'discriminator_stability'},
            'execution_backend': 'cuda',
            'source': {'digest': 'src-digest', 'origin_commit': 'a' * 40},
        }),
        ('02_diagnostic', {
            'candidate': {'id': 'diag-only-cand'},
            'view': {'id': 'discriminator_stability', 'evidence_scope': 'research_diagnostic'},
            'execution_backend': 'cuda',
            'source': {'digest': 'src-digest', 'origin_commit': 'a' * 40},
        }),
        ('03_other_commit', {
            'candidate': {'id': 'other-commit-cand'},
            'view': {'id': 'discriminator_stability'},
            'execution_backend': 'cuda',
            'source': {'digest': 'src-digest', 'origin_commit': 'b' * 40},
        }),
    ]:
        atomic_json(attempts / name / 'request.json', {'request': request})

    studies = [{
        'request': {
            'candidate': {'id': 'study-cand'},
            'view': {'id': 'discriminator_stability'},
            'execution_backend': 'cuda',
        }
    }]
    measured = publication._measured_candidates_at_source(
        tmp_path, manifests, studies, 'discriminator_stability', 'cuda'
    )
    assert measured == {'study-cand', 'non-study-cand'}


def test_comparison_projection_accepts_newly_registered_family_and_rejects_unregistered_or_incomplete():
    actual, projected, comparison = comparison_fixture()
    comparison['policy']['assignments'] = [
        {'task': 'acquire', 'qualification_tier': 1},
        {'task': 'held', 'qualification_tier': 2},
        {'task': 'hold', 'qualification_tier': 3},
    ]
    comparison['added_actual_tasks'] = ['added_tier1']
    comparison['reference_task_contracts'] = {}
    comparison['new_registered_families'] = ['rhbi']
    actual['rows'][0]['bindings']['trainer_family'] = 'rhbi'
    actual['rows'][0]['bindings']['task_contracts']['added_tier1'] = 'added-contract'
    actual['rows'][0]['tasks'].append({'task_id': 'added_tier1', 'status': 'PASS'})
    projected['rows'][0]['bindings']['trainer_family'] = 'rhbi'
    projected['rows'][0]['bindings']['task_contracts']['added_tier1'] = 'added-contract'

    assert publication._comparison_contracts_match(projected['rows'][0], comparison) is True
    publication._comparison_projection(actual, projected, comparison)
    assert any(t['task_id'] == 'added_tier1' and t['status'] == 'PASS'
               for t in projected['rows'][0]['nonrequired_tasks'])

    # Unregistered family without prior reference must fail closed.
    unregistered = deepcopy(comparison)
    unregistered['new_registered_families'] = []
    assert publication._comparison_contracts_match(projected['rows'][0], unregistered) is False

    # Missing required task or unexpected extra task must fail closed.
    missing_task = deepcopy(projected['rows'][0])
    missing_task['bindings']['task_contracts'].pop('acquire')
    assert publication._comparison_contracts_match(missing_task, comparison) is False
    extra_task = deepcopy(projected['rows'][0])
    extra_task['bindings']['task_contracts']['sneaky'] = 'x'
    assert publication._comparison_contracts_match(extra_task, comparison) is False


def test_calibrated_grid_triad_projection_accepts_fail_and_rejects_sub097_pass():
    actual, projected, comparison = comparison_fixture()
    comparison['calibrated_grid_triad_tasks'] = ['held']
    comparison['verified_evaluator_lineage'] = {'family': {'held': 'b-cal'}}
    actual['rows'][0]['bindings']['task_contracts']['held'] = 'b-cal'
    projected['rows'][0]['bindings']['task_contracts']['held'] = 'b-cal'

    # 1. FAIL under calibrated min_precision=0.75 is strictly FAIL under Rev-9 (0.97).
    publication._comparison_projection(deepcopy(actual), deepcopy(projected), comparison)

    # 2. PASS with precision >= 0.97 satisfies Rev-9 (0.97).
    pass_actual = deepcopy(actual)
    pass_projected = deepcopy(projected)
    pass_actual['rows'][0]['nonrequired_tasks'][0] = {'task_id': 'held', 'status': 'PASS', 'metrics': {'precision': 0.98}}
    pass_projected['rows'][0]['tasks'][-1] = {'task_id': 'held', 'status': 'PASS', 'metrics': {'precision': 0.98}}
    publication._comparison_projection(pass_actual, pass_projected, comparison)

    # 3. PASS with 0.75 <= precision < 0.97 must fail closed under Rev-9 projection.
    sub_actual = deepcopy(actual)
    sub_projected = deepcopy(projected)
    sub_actual['rows'][0]['nonrequired_tasks'][0] = {'task_id': 'held', 'status': 'PASS', 'metrics': {'precision': 0.82}}
    sub_projected['rows'][0]['tasks'][-1] = {'task_id': 'held', 'status': 'PASS', 'metrics': {'precision': 0.82}}
    with pytest.raises(ValueError, match='cannot grant Rev-9 PASS under calibrated min_precision=0.75'):
        publication._comparison_projection(sub_actual, sub_projected, comparison)




