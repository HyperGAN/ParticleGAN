"""Validated navigation for separate source-bound measurement cohorts.

These compact projections never become parent qualification inputs. Original
certified receipts and independently reconstructed reports own their verdicts.
"""
from pathlib import Path

from .contracts import file_hash, identifier, read_json, stable_hash
from .trainer_families import scientific_row_hash
from .views import load_view

REGISTRY = Path("reports/forge/scoped-publications.json")
SCHEMA = "forge_scoped_publications_v1"
HF_MEDIA_MANIFEST = Path("reports/forge/hf-media-manifest-2026-11-21.json")
HF_MEDIA_MANIFEST_SCHEMA = "forge_hf_media_manifest_v1"

_HF_MANIFEST_CACHE: dict[tuple[str, int, int, int], dict] = {}


def _relative_path(root, value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("publication artifacts require repository-relative paths")
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("publication artifact is missing or outside the repository")
    return path


def _relative(root, value):
    path = _relative_path(root, value)
    if not (root / path).is_file():
        raise ValueError("publication artifact is missing or outside the repository")
    return path


def _load_hf_media_manifest(root: Path) -> dict | None:
    manifest_path = Path(root) / HF_MEDIA_MANIFEST
    if not manifest_path.is_file():
        return None
    stat = manifest_path.stat()
    cache_key = (str(manifest_path.resolve()), stat.st_ino, stat.st_mtime_ns, stat.st_size)
    cached = _HF_MANIFEST_CACHE.get(cache_key)
    if cached is not None:
        return cached
    manifest = read_json(manifest_path)
    prefix = manifest.get("url_prefix")
    if (manifest.get("schema") != HF_MEDIA_MANIFEST_SCHEMA
            or not isinstance(prefix, str) or not prefix.startswith("https://")
            or not isinstance(manifest.get("files"), dict)):
        raise ValueError("unsupported HF media manifest")
    _HF_MANIFEST_CACHE[cache_key] = manifest
    return manifest


def hf_media_url(root: Path, relative_posix: str, expected_sha256: str) -> str | None:
    """Resolve an evicted Forge media asset to its immutable pinned HF URL offline."""
    manifest = _load_hf_media_manifest(Path(root))
    if manifest is None:
        return None
    entry = manifest["files"].get(relative_posix)
    if entry is None:
        return None
    if not isinstance(entry, list) or len(entry) != 3 or entry[1] != expected_sha256:
        raise ValueError("actual-training GIF changed after registration")
    if not relative_posix.startswith("reports/forge/"):
        raise ValueError("HF media manifest entry must be under reports/forge/")
    return manifest["url_prefix"] + relative_posix[len("reports/forge/"):]


def load_publications(root, load=None):
    """Return validated whole scoped rows and actual-training media bindings."""
    root = Path(root)
    load = load or (lambda path: read_json(root / path))
    published = {"cohorts": [], "media": {}, "clock_audits": {}, "final_measurements": {}}
    if not (root / REGISTRY).is_file():
        return published
    registry = load(REGISTRY)
    if (registry.get("schema") != SCHEMA or registry.get("qualification_input") is not False
            or not isinstance(registry.get("cohorts"), list)):
        raise ValueError("unsupported scoped publication registry")
    cohorts, media = published["cohorts"], published["media"]
    for entry in registry["cohorts"]:
        path = _relative(root, entry["path"])
        if file_hash(root / path) != entry["sha256"]:
            raise ValueError("scoped publication changed after registration")
        document = load(path)
        if (document.get("schema") != "forge_tier1_scoped_evidence_v1"
                or document.get("qualification_input") is not False
                or document.get("parent_qualification_credit") is not False):
            raise ValueError("scoped publication cannot grant parent qualification")
        if entry["view"] != document["view"]:
            raise ValueError("scoped publication view differs from its registry entry")
        view = load_view(root, document["view"])
        if view.get("reporting", {}).get("family_totals") is not False:
            raise ValueError("scoped publication requires a separate reporting cohort")
        if stable_hash(view) != document["policy_fingerprint"]:
            raise ValueError("scoped publication view differs from its frozen policy")
        if any(digest != stable_hash(contract) for digest, contract in document["task_contracts"].items()):
            raise ValueError("scoped publication changed its task contract catalog")
        for family, row in document["rows"].items():
            if (row["trainer_family"] != family
                    or scientific_row_hash(row) != document["scientific_rows_sha256"][family]
                    or row.get("bindings", {}).get("source_digest") != document["source_digest"]):
                raise ValueError("scoped publication changed its whole scientific row")
            for attempt in row.get("attempt_ids", []):
                identifier(attempt, "attempt")
                summary_path = Path("reports/forge/technique-receipts") / (attempt + ".json")
                summary = load(summary_path)
                if (summary.get("candidate_id") != row["candidate_id"]
                        or summary.get("candidate_revision") != row.get("candidate_revision")
                        or summary.get("provenance", {}).get("source_digest") != document["source_digest"]
                        or summary.get("provenance", {}).get("canonical_result_hash")
                        != document["receipt_result_hashes"].get(attempt)):
                    raise ValueError("scoped receipt differs from its recorded configuration/source")
            cohorts.append({"family": family, "row": row, "path": path.as_posix(),
                            "view": document["view"], "source_commit": document["source_commit"],
                            "task_contracts": document["task_contracts"],
                            "status_reasons": document.get("status_reasons", {})})
        for measured in document.get("final_measurements", []):
            attempt = identifier(measured["attempt_id"], "attempt")
            summary = load(Path("reports/forge/technique-receipts") / (attempt + ".json"))
            results = [result for result in summary["task_results"] if result["task_id"] == measured["task_id"]]
            if (summary["candidate_id"] != measured["candidate_id"]
                    or summary["candidate_revision"] != measured["candidate_revision"]
                    or summary["provenance"]["source_digest"] != measured["source_digest"]
                    or summary["provenance"]["canonical_result_hash"] != measured["canonical_result_hash"]
                    or len(results) != 1 or results[0]["gate_status"] != measured["status"]
                    or results[0]["compatibility_key"] != measured["compatibility_key"]
                    or measured["source_digest"] != document["source_digest"]
                    or document["receipt_result_hashes"].get(attempt) != measured["canonical_result_hash"]):
                raise ValueError("final measurement differs from its certified compact receipt")
            key = measured["family"], measured["task_id"]
            if key in published["final_measurements"]:
                raise ValueError("duplicate final measurement identity")
            published["final_measurements"][key] = measured
        for audit in document.get("clock_audits", []):
            key = audit["family"], audit["task_id"]
            measured = published["final_measurements"].get(key)
            if (not measured or audit["attempt_id"] != measured["attempt_id"]
                    or audit["canonical_result_hash"] != measured["canonical_result_hash"]
                    or audit["recorded_grade"] != measured["status"]
                    or audit.get("qualification_input") is not False
                    or set(audit["comparisons"]) != {"step_label", "horizon", "evaluation_cadence", "restart"}
                    or any(check["digest_equal"] != (check["reference_sha256"] == check["changed_sha256"])
                           for check in audit["comparisons"].values())
                    or audit["clock_dependency_count"] != len(audit["unexplained_clock_dependencies"])):
                raise ValueError("clock display evidence differs from the final measured attempt")
            published["clock_audits"][key] = {**audit, "path": path.as_posix()}
    media_reference = registry.get("media")
    if media_reference:
        path = _relative(root, media_reference["path"])
        if file_hash(root / path) != media_reference["sha256"]:
            raise ValueError("actual-training media index changed after registration")
        index = load(path)
        if index.get("qualification_input") is not False or not isinstance(index.get("items"), list):
            raise ValueError("unsupported actual-training media index")
        for item in index["items"]:
            gif = _relative_path(root, item["gif"])
            if (root / gif).is_file():
                if file_hash(root / gif) != item["gif_sha256"]:
                    raise ValueError("actual-training GIF changed after registration")
            elif hf_media_url(root, gif.as_posix(), item["gif_sha256"]) is None:
                raise ValueError("publication artifact is missing or outside the repository")
            if (item.get("kind") != "actual_training_saved_observations_gif"
                    or item.get("optimizer_updates_added") != 0 or item.get("sampling_draws_added") != 0):
                raise ValueError("media must use the recorded training observations without new execution")
            sidecar = _relative(root, gif.with_suffix(".json").as_posix())
            if load(sidecar) != {key: value for key, value in item.items()
                                  if key not in {"family", "attempt_id", "gif"}}:
                raise ValueError("actual-training media index differs from its provenance receipt")
            key = item["family"], item["task_id"], item["attempt_id"]
            if key in media:
                raise ValueError("duplicate actual-training media identity")
            media[key] = item
    for (family, task), measured in published["final_measurements"].items():
        item = media.get((family, task, measured["attempt_id"]))
        if not item or item["recorded_grade"] != measured["status"]:
            raise ValueError("final measurement lacks its exact actual-training GIF")
    return published


def clock_audit_summary(root, measured):
    """Project four certified digest comparisons; never regrade the clock gate."""
    root = Path(root)
    original = read_json(root / "reports/forge/attempts" / measured["attempt_id"] / "result.json")
    if stable_hash(original) != measured["canonical_result_hash"]:
        raise ValueError("clock original differs from its certified canonical result")
    result = next(row for row in original["task_results"] if row["task_id"] == measured["task_id"])
    evidence = result.get("evidence", {})
    comparisons = evidence.get("comparisons")
    if comparisons is None:
        return None
    names = {"step_label", "horizon", "evaluation_cadence", "restart"}
    if (len(comparisons) != 4 or {row.get("condition") for row in comparisons} != names
            or not isinstance(evidence.get("source_audit"), dict)):
        raise ValueError("clock display summary requires all four certified comparisons and source audit")
    audit = evidence["source_audit"]
    dependencies = audit["unexplained_clock_dependencies"]
    return {"family": measured["family"], "task_id": measured["task_id"],
            "attempt_id": measured["attempt_id"], "canonical_result_hash": measured["canonical_result_hash"],
            "recorded_grade": result["gate_status"], "qualification_input": False,
            "comparisons": {row["condition"]: {"reference_sha256": row["reference_sha256"],
                            "changed_sha256": row["changed_sha256"],
                            "digest_equal": row["reference_sha256"] == row["changed_sha256"]}
                            for row in comparisons},
            "clock_dependency_count": len(dependencies), "unexplained_clock_dependencies": dependencies,
            "source_audit": {key: audit[key] for key in ("source_sha256", "allowed_state", "scope")}}
