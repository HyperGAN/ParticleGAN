"""Bind future task executions to repaired plumbing, preserving old evidence.

Run after provider/adapter fixes settle. This never grades or executes a task.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = "92b565b09"
PROVIDERS = [f"benchmarks/toy_audit/api_model_{name}.py" for name in ("ae", "ddgan", "vae")]
ALLOWED = {*PROVIDERS, "experiments/forge/model_toy_adapter.py", "benchmarks/toy_audit/api_images.py"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def original(path):
    return subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)


def declarations(data):
    result = {}
    for node in ast.parse(data).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and method.name != "__init__":
                    result[f"{node.name}.{method.name}"] = ast.dump(method, include_attributes=False)
    return result


def main():
    for source in PROVIDERS:
        assert declarations(original(source)) == declarations((ROOT / source).read_bytes()), source
    rows = []
    for path in sorted((ROOT / "configs/forge/tasks").glob("*.json")):
        if not (path.stem.startswith("model_") or path.stem == "five_word_joint_acquisition"):
            continue
        relative = str(path.relative_to(ROOT))
        before_bytes = original(relative)
        before = json.loads(before_bytes)
        after = json.loads(path.read_text())
        expected = json.loads(before_bytes)
        changes = []
        for source, pin in before["evaluation"].get("sources", {}).items():
            current = sha((ROOT / source).read_bytes())
            if current != pin:
                assert source in ALLOWED, f"unrelated source change: {source}"
                expected["evaluation"]["sources"][source] = current
                changes.append(dict(path=["evaluation", "sources", source], old_sha256=pin, new_sha256=current))
        assert after in (before, expected), f"non-source task mutation: {relative}"
        if not changes:
            continue
        preserved = HERE / "previous-task-definitions-infra-v2" / path.name
        preserved.parent.mkdir(exist_ok=True)
        if preserved.exists():
            assert preserved.read_bytes() == before_bytes
        else:
            preserved.write_bytes(before_bytes)
        path.write_text(json.dumps(expected, indent=2) + "\n")
        rows.append(dict(task=path.stem, original_path=relative,
                         original_sha256=sha(before_bytes), future_sha256=sha(path.read_bytes()),
                         original_blob=subprocess.check_output(["git", "rev-parse", f"{BASE}:{relative}"], cwd=ROOT, text=True).strip(),
                         preserved_original=str(preserved.relative_to(ROOT)), changes=changes))
    receipt = dict(schema_version=1, scope="future_task_source_binding_only",
                   original_commit=subprocess.check_output(["git", "rev-parse", BASE], cwd=ROOT, text=True).strip(),
                   unchanged_provider_methods_except_optimizer_construction=True,
                   task_architecture_data_prior_initialization_budget_sampling_cadence_gates_unchanged=True,
                   original_verdicts_preserved=True, qualification_recomputed=False,
                   training_updates=0, random_draws=0, tasks=rows)
    (HERE / "infrastructure-source-binding.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(tasks=len(rows), training_updates=0, verdicts_changed=False)))


if __name__ == "__main__":
    main()
