"""Provider discovery must work with only declared, immutable snapshot files."""
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.forge.model_toy_sources import model_toy_source_paths
from experiments.forge.sources import inspect_source, snapshot_source

ROOT = Path(__file__).resolve().parents[1]


def test_ae_discovery_reads_frozen_resources_without_checkout_fallback(tmp_path):
    task = json.loads((ROOT / "configs/forge/tasks/model_ae_ddgan_gaussian_chain.json").read_text())
    resources = model_toy_source_paths(task)
    manifest = inspect_source(ROOT, resources)
    frozen = snapshot_source(ROOT, tmp_path / "queue", manifest)
    script = """
import json
from pathlib import Path
from benchmarks.toy_audit import api_model_ae as provider
assert Path(provider.__file__).resolve().is_relative_to(Path.cwd())
cases = provider.list_cases()
print(json.dumps(sorted(case['id'] for case in cases)))
"""
    env = {**os.environ, "PYTHONPATH": str(frozen), "OMP_NUM_THREADS": "1"}
    result = subprocess.run([sys.executable, "-c", script], cwd=frozen, env=env,
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == ["api-model-ae-gan-multimodal", "api-model-ae-reconstruction"]
    for relative in resources:
        assert (frozen / relative).read_bytes() == (ROOT / relative).read_bytes()
