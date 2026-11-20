"""Reject new Ruff findings while retaining reviewed legacy fixtures and exports.

Use --update-baseline after reviewing additions or resolving existing findings.
The compact baseline stores messages and counts, without volatile line numbers.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PATHS = ("particlegan", "particlegan_forge", "examples", "lib", "experiments",
         "benchmarks", "scripts", "ops", "tests")
RULES = "F,E9"
# Undefined names cannot be accepted by refreshing the legacy baseline. Reviewed
# F811 entries are imported pytest fixtures shadowed by injected fixture arguments;
# they remain counted exceptions, so a new redefinition still fails the gate.
STRICT_CODES = frozenset({"F821"})
# These exact frozen sources have reviewed control-flow false positives. The
# proximal replay reads ``restored`` at update 33 after update 32 initialized it;
# caption closures read ``p`` before its end-of-arm deletion. Their original
# source hashes are part of experiment contracts, so do not edit them for lint.
# A changed byte, different message or new undefined name is never suppressed.
FROZEN_F821_FALSE_POSITIVES = {
    "benchmarks/toy_audit/prior_proximal_smoke.py": (
        "e1f78fc72d6bcda99aa56e20dcf1e4ec327dda668308afb599d9b30e9903edd4",
        "Undefined name `restored`"),
    "examples/e22_routed_caption_late_phase.py": (
        "f4bad83b5ff9944a14c75bcf5d51d82e26fd3ed432a0730122255fffdc313b65",
        "Undefined name `p`"),
    "examples/e22_routed_caption_untied.py": (
        "a35938f51b7974677651d2c05b1aa396896255460e213cf26ccf57409efcdb81",
        "Undefined name `p`"),
}


def _ruff_command() -> list[str]:
    if importlib.util.find_spec("ruff") is not None:
        return [sys.executable, "-m", "ruff"]
    binary = shutil.which("ruff")
    if binary:
        return [binary]
    return [sys.executable, "-m", "ruff"]


def findings(root: Path) -> Counter:
    result = subprocess.run(
        [*_ruff_command(), "check", "--select", RULES,
         "--output-format", "json", *PATHS],
        cwd=root, capture_output=True, text=True,
    )
    if result.returncode not in (0, 1) or not result.stdout.strip():
        raise RuntimeError(result.stderr.strip() or "Ruff failed")
    counts = Counter()
    for item in json.loads(result.stdout):
        path = Path(item["filename"]).resolve().relative_to(root.resolve()).as_posix()
        message = re.sub(r"from line \d+", "from an earlier line", item["message"])
        frozen = FROZEN_F821_FALSE_POSITIVES.get(path)
        if (item["code"] == "F821" and frozen is not None and message == frozen[1]
                and hashlib.sha256((root / path).read_bytes()).hexdigest() == frozen[0]):
            continue
        counts[(path, item["code"], message)] += 1
    return counts


def unpack(document: dict) -> Counter:
    if document.get("schema_version") != 1 or document.get("rules") != RULES:
        raise ValueError("unsupported lint baseline; review and regenerate it")
    return Counter({(path, code, message): count
                    for path, rules in document["findings"].items()
                    for code, messages in rules.items()
                    for message, count in messages.items()})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    baseline = root / ".ruff-baseline.json"
    current = findings(root)
    strict = {key: count for key, count in current.items() if key[1] in STRICT_CODES}
    if strict:
        for (path, code, message), count in sorted(strict.items()):
            print(f"STRICT: {path}: {code} {message} ({count})")
        print("Undefined names must be fixed; they cannot enter the lint baseline")
        return 1
    if args.update_baseline:
        grouped = {}
        for (path, code, message), count in sorted(current.items()):
            grouped.setdefault(path, {}).setdefault(code, {})[message] = count
        baseline.write_text(json.dumps({"schema_version": 1, "rules": RULES,
                                       "findings": grouped}, indent=2, sort_keys=True) + "\n")
        print(f"Recorded {sum(current.values())} reviewed Ruff findings in {baseline.name}")
        return 0
    expected = unpack(json.loads(baseline.read_text()))
    if any(code in STRICT_CODES for _, code, _ in expected):
        print("The lint baseline contains a strict code; review and remove its exemption")
        return 1
    added, removed = current - expected, expected - current
    for label, delta in (("NEW", added), ("RESOLVED", removed)):
        for (path, code, message), count in sorted(delta.items()):
            print(f"{label}: {path}: {code} {message} ({count})")
    if added or removed:
        print("Review the changes, then run python scripts/check_lint.py --update-baseline")
        return 1
    print(f"Lint baseline current: {sum(current.values())} existing findings; no additions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
