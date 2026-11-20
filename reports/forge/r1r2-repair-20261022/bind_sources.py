"""Refresh future task source pins after the opt-in word trainer addition.

Preserve original task bytes and scoring/data definitions. This performs no
execution, grading, queue admission or qualification recomputation.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = 'benchmarks/toy_audit/api_images.py'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_bytes(commit, path):
    return subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT)


def main():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    prior_source = git_bytes(commit, SOURCE)
    current_source = (ROOT / SOURCE).read_bytes()
    before_ast, after_ast = ast.parse(prior_source), ast.parse(current_source)
    before_functions = {node.name: ast.dump(node, include_attributes=False)
                        for node in before_ast.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    after_functions = {node.name: ast.dump(node, include_attributes=False)
                       for node in after_ast.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert before_functions == after_functions, 'source binding cannot conceal a changed data/scoring function'
    assert {'word_bank', 'score_words', '_word_metadata'} <= before_functions.keys()
    before_classes = {node.name: ast.dump(node, include_attributes=False)
                      for node in before_ast.body if isinstance(node, ast.ClassDef)}
    after_classes = {node.name: ast.dump(node, include_attributes=False)
                     for node in after_ast.body if isinstance(node, ast.ClassDef)}
    changed_classes = sorted(name for name in before_classes if before_classes[name] != after_classes.get(name))
    assert set(changed_classes) <= {'WordFixture', 'WordGenerator'}
    prior_dir = HERE / 'previous-task-definitions'
    prior_dir.mkdir(exist_ok=True)
    view_path = ROOT / 'configs/forge/views/discriminator_stability.json'
    view = json.loads(view_path.read_text())
    rows = []
    for assignment in view['assignments']:
        relative = f"configs/forge/tasks/{assignment['task']}.json"
        original_bytes = git_bytes(commit, relative)
        original = json.loads(original_bytes)
        if SOURCE not in original['evaluation'].get('sources', {}):
            continue
        preserved = prior_dir / f"{assignment['task']}.json"
        if preserved.exists():
            assert preserved.read_bytes() == original_bytes, 'original task definition changed'
        else:
            preserved.write_bytes(original_bytes)
        task_path = ROOT / relative
        current = json.loads(task_path.read_text())
        baseline = json.loads(json.dumps(original))
        baseline['evaluation']['sources'][SOURCE] = current['evaluation']['sources'][SOURCE]
        assert current == baseline, 'only the future execution source hash may change'
        current['evaluation']['sources'][SOURCE] = sha(current_source)
        task_path.write_text(json.dumps(current, indent=2) + '\n')
        rows.append(dict(task=assignment['task'], importance=assignment['importance'],
            qualification_tier=assignment['qualification_tier'],
            original_task_path=relative, original_task_sha256=sha(original_bytes),
            original_task_blob=subprocess.check_output(['git', 'rev-parse', f'{commit}:{relative}'], cwd=ROOT, text=True).strip(),
            preserved_original=str(preserved.relative_to(ROOT)),
            future_task_sha256=sha(task_path.read_bytes()),
            old_source_pin=original['evaluation']['sources'][SOURCE],
            new_source_pin=sha(current_source), changed_paths=[['evaluation', 'sources', SOURCE]]))
    result = dict(schema_version=1, scope='future_task_source_binding_only',
        original_commit=commit, view='discriminator_stability', view_revision=view['revision'],
        view_sha256=sha(view_path.read_bytes()), source_path=SOURCE,
        original_source_sha256=sha(prior_source), future_source_sha256=sha(current_source),
        unchanged_top_level_functions=sorted(before_functions), changed_classes=changed_classes,
        preserved_data_architecture_prior_initialization_budget_sampling_cadence_gates=True,
        original_verdicts_preserved=True, qualification_recomputed=False,
        training_updates=0, random_draws=0, tasks=rows)
    (HERE / 'source-binding-migration.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(tasks=len(rows), source_sha256=sha(current_source),
                         verdicts_changed=False, training_updates=0)))


if __name__ == '__main__':
    main()
