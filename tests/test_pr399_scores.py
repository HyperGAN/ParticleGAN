"""Original source identities and full gates survive PR399's retirement."""
import json
from pathlib import Path

import pytest

from pr_integration_compat import before_pr_integration
from experiments.forge.pr399_scores import PINS, publications, tier_rows, family_lines
from experiments.forge.source_bound_board import BOARD_SOURCE, validate_board
from experiments.forge.test_performance import SUPPLEMENTAL, build_report

ROOT = Path(__file__).resolve().parents[1]


def requirements():
    board = validate_board(json.loads((ROOT / BOARD_SOURCE).read_text()))
    return {str(tier): {a['task'] for a in board['recorded_view']['assignments']
                       if a['importance'] == 'required' and a['qualification_tier'] == tier}
            for tier in (1, 2, 3)}


def test_all_42_original_attempts_are_indexed_and_separate_from_qualification():
    packets = publications(ROOT)
    assert [len(d['cells']) for _, d in packets] == [21, 3, 12, 6]
    report = build_report(ROOT)
    attempts = {c['result']['attempt_id'] for _, d in packets for c in d['cells']}
    indexed = [r for r in report['records'].values()
               if r.get('result', {}).get('attempt_id') in attempts]
    assert len(indexed) == 42
    assert all(r['receipt'] in PINS and r['kind'] == 'optimizer' for r in indexed)
    assert all(d['qualification_input'] is False and d['qualification_reuse'] is False for _, d in packets)


def test_tier_rows_keep_whole_recipe_source_and_gate_incompleteness():
    rows = tier_rows(ROOT, ROOT / 'reports/forge/technique-inventory.md', requirements())
    assert set(rows) == {'bcap-pure'}  # Atlas policy diagnostics are not MoG gauntlet credit.
    assert len(rows['bcap-pure']) == 12  # Ten distinct successor recipes/sources and two paired arms.
    candidate = next(r for r in rows['bcap-pure'] if 'generator-idle-ema-bcap' in r)
    control = next(r for r in rows['bcap-pure'] if 'bcap-default-baseline-direction' in r)
    # Counts name every unfinished cell. Same-source backfill attempts from
    # reports/forge/attempt-coverage.json fill only the originally unrun cells.
    assert '[4/6 (2 incomplete)]' in candidate and '[12/29 (3 blocked, 2 incomplete)]' in candidate
    assert '[5/6 (1 incomplete)]' in control and '[13/29 (2 blocked, 1 error, 1 incomplete)]' in control
    # Two source-bound Gaussian retention wins do not pool their unequal-mass failures.
    history = next(r for r in rows['bcap-pure'] if 'kinetic-local-dualnorm-g-history-v1;' in r)
    assert '[2/6 (1 timeout)]' in history and '[9/21 (1 blocked)]' in history
    assert '[11/29 (2 blocked, 1 timeout)]' in history


def test_successor_and_paired_task_semantics_match_the_recorded_ordinary_questions():
    for path, document in publications(ROOT):
        if '-atlas-' in path:
            continue
        for condition in document['condition_contracts'].values():
            original = condition['task']
            declared = json.loads((ROOT / 'configs/forge/tasks' / (original['id'] + '.json')).read_text())
            declared = before_pr_integration(
                ROOT, declared, "configs/forge/tasks/" + original["id"] + ".json")
            assert original['execution'] == declared['execution']
            # Preserve original source and evaluator identities separately;
            # only source hashes and their revision explanations differ.
            for section in ('evaluation',):
                def scientific(value):
                    return {k: v for k, v in value.items() if k not in {'sources', 'evaluator_revision'}}
                assert scientific(original[section]) == scientific(declared[section])


def test_family_details_retain_failure_metrics_costs_and_archived_curve_pins():
    bcap = '\n'.join(family_lines(ROOT, ROOT / 'reports/forge/families/bcap-pure.md', 'bcap-pure'))
    atlas = '\n'.join(family_lines(ROOT, ROOT / 'reports/forge/families/atlas.md', 'atlas'))
    assert '**4/6 Tier 1**' in bcap and '**5/6**' in bcap
    assert '946/1,000' in bcap and '12,613/20,001' in bcap and 'terminal update count is unknown' in bcap
    assert '2,052.616064' in bcap and '2,072.815230' in bcap
    assert '**1 PASS / 1 FAIL / 4 INCOMPLETE**' in atlas
    for _, document in publications(ROOT):
        for cell in document['cells']:
            diag = cell['result'].get('saved_sample_diagnostic') or {}
            assert 'observations' not in diag
            if diag.get('archived_observations_count'):
                assert len(diag['archived_observations_sha256']) == 64


def test_changed_compact_grade_or_missing_index_cannot_render(tmp_path):
    registry = json.loads((ROOT / SUPPLEMENTAL).read_text())
    (tmp_path / SUPPLEMENTAL).parent.mkdir(parents=True)
    for path in PINS:
        (tmp_path / path).write_bytes((ROOT / path).read_bytes())
    (tmp_path / SUPPLEMENTAL).write_text(json.dumps(registry))
    path = next(iter(PINS))
    changed = json.loads((tmp_path / path).read_text())
    changed['cells'][0]['status'] = 'FAIL'
    (tmp_path / path).write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='evidence bytes changed'):
        publications(tmp_path)
    (tmp_path / path).write_bytes((ROOT / path).read_bytes())
    registry['publications'] = [pin for pin in registry['publications'] if pin['path'] != path]
    (tmp_path / SUPPLEMENTAL).write_text(json.dumps(registry))
    with pytest.raises(ValueError, match='missing from display index'):
        publications(tmp_path)
