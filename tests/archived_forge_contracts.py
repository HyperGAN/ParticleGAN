"""Exact archived declarations for software controls, never current qualification.

The fixtures preserve original UTF-8 bytes and Git blob identities. Tests replay
the published checkout's contracts in private directories; live declarations,
published selections and saved evidence are never rewritten.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/archived-forge-contracts.json"
COMMITS = {
    "published_develop": "5737ade47dca89b04d338ada20667781d1f2b5df",
    "published_develop_views": "5737ade47dca89b04d338ada20667781d1f2b5df",
    "tier1_completion": "928b485ffbe6e17d79b41d307ae8b6275489b37a",
    "shallow_gaussian": "1853cfdaf31c243c8a1e75a86f370ba143691d5a",
    "shallow_gaussian_cards": "5737ade47dca89b04d338ada20667781d1f2b5df",
    "bcap_legacy_questions": "5737ade47dca89b04d338ada20667781d1f2b5df",
    "bcap_prior_baseline_questions": "03466efa4de8271b9a2c964406dc6f4f0f260792",
    "model_toy_api_callers": "bee9f06ed21695e518c1cb702e12a385ab9389e8",
}


_FIXTURE_CACHE: dict[tuple[int, int, int], dict] = {}
_VERIFIED_BYTES_CACHE: dict[tuple, bytes] = {}


def _load_fixture_packet():
    st = FIXTURE.stat()
    key = (st.st_ino, st.st_mtime_ns, st.st_size)
    packet = _FIXTURE_CACHE.get(key)
    if packet is None:
        packet = json.loads(FIXTURE.read_text())
        assert packet["schema"] == "forge_archived_software_contract_fixtures_v1"
        assert packet["qualification_input"] is False
        _FIXTURE_CACHE.clear()
        _FIXTURE_CACHE[key] = packet
    return key, packet


def archived_contract_bytes(scope, relative):
    fkey, packet = _load_fixture_packet()
    cohort = packet[scope]
    assert cohort["source_commit"] == COMMITS[scope]
    record = cohort["files"][relative]
    if record.get("repository_bytes") is True:
        src = ROOT / relative
        st = src.stat()
        bkey = (fkey, scope, relative, st.st_ino, st.st_mtime_ns, st.st_size)
        cached = _VERIFIED_BYTES_CACHE.get(bkey)
        if cached is not None:
            return cached
        data = src.read_bytes()
    else:
        bkey = (fkey, scope, relative)
        cached = _VERIFIED_BYTES_CACHE.get(bkey)
        if cached is not None:
            return cached
        data = record["text"].encode("utf-8")
    assert hashlib.sha256(data).hexdigest() == record["sha256"]
    assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == record["git_blob"]
    _VERIFIED_BYTES_CACHE[bkey] = data
    return data


def restore_archived_contracts(root, scope):
    """Restore only private fixture bytes, with an explicit immutable origin."""
    root = Path(root).resolve()
    assert root != ROOT.resolve()
    _, packet = _load_fixture_packet()
    for relative in packet[scope]["files"]:
        path = root / relative
        assert path.resolve().is_relative_to(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archived_contract_bytes(scope, relative))
    metrics_path = root / "benchmarks/toy100/metrics.py"
    if scope in {"bcap_prior_baseline_questions", "tier1_completion"} and metrics_path.is_file():
        metrics_path.write_text(
            metrics_path.read_text().replace(
                "MIN_PRECISION = 0.75  # calibrated clean-sampling gate (oracle: 0.98889, filament mass: 0.20-0.25)",
                "MIN_PRECISION = 0.97  # oracle value: 1 - exp(-9/2) = 0.98889",
            )
        )
        tasks_dir = root / "configs/forge/tasks"
        if scope == "tier1_completion" and tasks_dir.is_dir():
            for task_path in tasks_dir.glob("*.json"):
                card = json.loads(task_path.read_text())
                rev = card.get("evaluation", {}).get("evaluator_revision") or {}
                if rev.get("id") == "tier2-grid-triad-clean-sampling-calibration-v1":
                    card["evaluation"]["coverage_thresholds"]["min_precision"] = 0.97
                    task_path.write_text(json.dumps(card, indent=2) + "\n")
    return root


def restore_published_view_roster(root):
    """Replay exactly the views declared by the original published report."""
    root = Path(root).resolve()
    assert root != ROOT.resolve()
    packet = json.loads(FIXTURE.read_text())
    originals = set(packet["published_develop_views"]["files"])
    directory = root / "configs/forge/views"
    assert directory.resolve().is_relative_to(root)
    for declaration in directory.glob("*.json"):
        assert declaration.resolve().is_relative_to(root)
        if declaration.relative_to(root).as_posix() not in originals:
            declaration.unlink()
    return restore_archived_contracts(root, "published_develop_views")


def commit_available(commit):
    """True when this checkout can read the archived commit (shallow clones cannot)."""
    return subprocess.run(["git", "cat-file", "-e", commit + "^{commit}"], cwd=ROOT,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def require_history(scope):
    """Skip archived-history controls with an explicit reason instead of erroring."""
    import pytest
    commit = COMMITS[scope]
    if not commit_available(commit):
        pytest.skip(f"archived {scope} commit {commit[:12]} is unavailable in this checkout "
                    "(shallow clone); fetch full history to run this control")


def published_develop_checkout(tmp_path):
    """Read original committed reports against their published develop cards."""
    # These report controls only read their inputs; share one complete physical
    # report mirror per pytest run so containment guards retain their meaning.
    require_history("published_develop")
    root = tmp_path.parent / "published-develop-checkout"
    complete = root.with_name(root.name + ".complete")
    if complete.is_file() and root.is_dir():
        return root
    if root.exists():
        # An earlier call failed part-way; never hand out a partial mirror.
        shutil.rmtree(root)
    root.mkdir()
    # Reports and artifact links are read-only inputs in these controls. Configs
    # are private copies, allowing exact original task identities to be restored.
    for path in ROOT.iterdir():
        if path.name not in {".git", "configs", "reports"}:
            destination = root / path.name
            if path.name in {"particlegan", "experiments", "benchmarks", "docs"}:
                shutil.copytree(path, destination)
            elif path.is_file():
                shutil.copyfile(path, destination)
            else:
                destination.symlink_to(path, target_is_directory=path.is_dir())
    shutil.copytree(ROOT / "configs", root / "configs")
    # The publication's recorded input manifest is sufficient for these
    # read-only controls. Do not copy unrelated research logs or checkpoints.
    commit = COMMITS["published_develop"]
    report_path = "reports/forge/technique-inventory.json"
    report_bytes = subprocess.check_output(["git", "show", f"{commit}:{report_path}"], cwd=ROOT)
    publication = json.loads(report_bytes)
    for relative, digest in publication["family_progress"]["input_hashes"].items():
        if not relative.startswith("reports/"):
            continue
        source, destination = ROOT / relative, root / relative
        data = source.read_bytes() if source.is_file() else None
        if data is None or hashlib.sha256(data).hexdigest() != digest:
            data = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        assert hashlib.sha256(data).hexdigest() == digest
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    restore_archived_contracts(root, "published_develop")
    restore_archived_contracts(root, "bcap_legacy_questions")
    # Current publication refreshes must not enter these historical controls.
    # Restore their original compact report and whole-selection inputs from
    # the exact immutable publication commit, without hydrating raw artifacts.
    for relative in ("reports/forge/technique-inventory.json",
                     "reports/forge/technique-inventory.md",
                     "configs/forge/selections/family-current-v1.json",
                     "configs/forge/trainer-families.json"):
        data = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        blob = subprocess.check_output(["git", "rev-parse", f"{commit}:{relative}"], cwd=ROOT,
                                       text=True).strip()
        assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == blob
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(data)
    # Documentation validates its local source and illustration targets too;
    # those navigation-only files are outside the numerical input manifest.
    family_ids = {row["trainer_family"] for row in
                  publication["rows"] + publication.get("historical_family_rows", [])}
    navigation = set()
    for family_id in family_ids:
        documentation = json.loads((root / "configs/forge/family-documentation" /
                                    (family_id + ".json")).read_text())
        navigation.update(path for path in documentation["sources"] if path.startswith("reports/"))
        illustration = documentation.get("illustration", {}).get("path", "")
        if illustration.startswith("reports/"):
            navigation.add(illustration)
    for relative in sorted(navigation):
        destination = root / relative
        if not destination.is_file():
            data = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    result = restore_published_view_roster(root)
    complete.write_text("complete\n")
    return result


def bcap_legacy_software_checkout(directory, *, code_source=ROOT):
    """Current code with exact original BCAP questions, only for software replay.

    The frozen comparator reads Git history but never writes it. Its private
    checkout therefore has a read-only history pointer; no admission helper
    may use this checkout. Current migrated questions are tested separately.
    """
    root = Path(directory).resolve()
    code_source = Path(code_source).resolve()
    assert root != ROOT.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in ("particlegan", "experiments", "benchmarks", "lib", "configs"):
        shutil.copytree(code_source / name, root / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # Frozen evaluators may import report-owned Python. No bulk logs, tensor
    # dumps, old metrics or qualification receipts enter this software replay.
    report_sources = list((code_source / "reports").rglob("*.py"))
    report_sources.append(code_source / "reports/transfer_suite/unadjusted/leading_profile.json")
    for source in report_sources:
        destination = root / source.relative_to(code_source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    restore_archived_contracts(root, "bcap_legacy_questions")
    api_path = root / "experiments/forge/api.py"
    if api_path.is_file():
        api_path.write_text(api_path.read_text().replace("            input_noise_std=0.5,\n", ""))
    # Current code has explicit source guards. Rebind only their byte identity;
    # preserve the original question and exact ancestry expected by the frozen
    # comparator's unchanged source-revision assertions.
    cohort = json.loads(FIXTURE.read_text())["bcap_legacy_questions"]
    rebinding = {}
    for relative in cohort["files"]:
        if "/tasks/" not in relative:
            continue
        path = root / relative
        card = json.loads(path.read_text())
        sources = card["evaluation"].get("sources", {})
        changed = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                   for name, old in sources.items() if (root / name).is_file()
                   and hashlib.sha256((root / name).read_bytes()).hexdigest() != old}
        if changed:
            previous = dict(sources)
            sources.update(changed)
            card["evaluation"]["evaluator_revision"] = {
                "id": "bcap-legacy-prior-software-source-binding-v1",
                "qualification_input": False,
                "original_task_commit": cohort["source_commit"],
                "original_task_path": relative,
                "previous_revision": card["evaluation"].get("evaluator_revision"),
                "previous_source_sha256": {name: previous[name] for name in changed},
            }
            path.write_text(json.dumps(card, indent=2) + "\n")
            rebinding[relative] = {name: {"original_sha256": previous[name],
                                        "current_sha256": digest}
                                  for name, digest in changed.items()}
    (root / "legacy-software-source-bindings.json").write_text(json.dumps({
        "qualification_input": False, "original_task_source_commit": cohort["source_commit"],
        "source_rebindings": rebinding}, indent=2) + "\n")
    gitdir = subprocess.check_output(["git", "rev-parse", "--absolute-git-dir"],
                                     cwd=ROOT, text=True).strip()
    (root / ".git").write_text("gitdir: " + gitdir + "\n")
    return root


def bcap_pinned_software_checkout(directory, commit):
    """Replay a declared code cohort with original BCAP question bytes.

    Only source/configuration files are extracted. No historical run, trained
    checkpoint, score or publication is overwritten or used as a new verdict.
    """
    import io
    import tarfile

    if not commit_available(commit):
        import pytest
        pytest.skip(f"pinned BCAP software source {commit} requires full Git history")
    directory = Path(directory).resolve()
    source = directory / "pinned-source"
    source.mkdir(parents=True, exist_ok=False)
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit, "--", "reports"], cwd=ROOT, text=True).splitlines()
    support = [name for name in names if name.endswith(".py")]
    data = subprocess.check_output(["git", "archive", commit, "--",
        "particlegan", "experiments", "benchmarks", "lib", "configs",
        "reports/transfer_suite/unadjusted/leading_profile.json", *support], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(data)) as bundle:
        bundle.extractall(source, filter="data")
    result = bcap_legacy_software_checkout(directory / "legacy-questions", code_source=source)
    (directory / "source-cohort.json").write_text(json.dumps({
        "code_source_commit": commit, "question_source_commit": COMMITS["bcap_legacy_questions"],
        "source_archive_sha256": hashlib.sha256(data).hexdigest(),
        "qualification_input": False, "scope": "two-update software compatibility",
    }, indent=2) + "\n")
    return result


def current_policy_software_checkout(tmp_path):
    """Original parent questions plus freshly bound public-policy source bytes."""
    from experiments.forge.planning import resolve_idea
    from experiments.forge.tier1_policy import write_declarations
    root = tmp_path / "policy-software-checkout"
    source = resolve_idea(ROOT, "k3p")["source"]
    for relative in source["files"]:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    shutil.copytree(ROOT / "configs/forge", root / "configs/forge", dirs_exist_ok=True)
    restore_archived_contracts(root, "published_develop")
    write_declarations(root)
    return root
