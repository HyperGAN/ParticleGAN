"""Data-driven laws registry validation and dynamic rendering guarantees."""
from pathlib import Path
import re

import pytest

from experiments.forge.contracts import atomic_json, read_json
from experiments.forge import laws_registry


ROOT = Path(__file__).resolve().parents[1]


def _github_heading_slug(text: str) -> str:
    slug = text.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = slug.replace(" ", "-")
    return slug


def _make_registry(tmp_path: Path, laws=None, **overrides):
    doc = tmp_path / "docs/custom-laws.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("# Custom Laws\n\n### Law Alpha — Geometry\n\n### Law Beta — Dynamics\n")
    payload = {
        "schema_version": 1,
        "epistemic_status": "provisional_evolving",
        "epigraph": "Test laws are provisional.",
        "epistemic_note": "Subject to revision as counterexamples emerge.",
        "doc_path": "docs/custom-laws.md",
        "laws": laws if laws is not None else [
            {
                "id": "law_alpha",
                "short_label": "Law Alpha",
                "title": "Geometry",
                "domain_level": "Geometry",
                "question": "Can the system represent the geometry?",
                "doc_anchor": "law-alpha--geometry",
                "status": "active",
                "sub_laws": [
                    {
                        "id": "law_alpha_1",
                        "short_label": "Law Alpha.1",
                        "title": "Sub-geometry",
                        "doc_anchor": "law-alpha--geometry",
                    }
                ],
            },
            {
                "id": "law_beta",
                "short_label": "Law Beta",
                "title": "Dynamics",
                "domain_level": "Dynamics",
                "question": "Does the coupled loop converge?",
                "doc_anchor": "law-beta--dynamics",
                "status": "provisional",
                "sub_laws": [],
            },
        ],
    }
    payload.update(overrides)
    atomic_json(tmp_path / laws_registry.LAWS_REGISTRY, payload)
    return payload


def test_committed_laws_registry_is_valid_and_matches_doc_anchors():
    summary = laws_registry.validate_laws_registry(ROOT)
    assert summary["status"] == "valid"
    assert summary["schema_version"] == 1
    assert summary["epistemic_status"] == "provisional_evolving"
    assert summary["laws"] >= 1
    data = laws_registry.load_laws(ROOT)
    assert data is not None
    doc_text = (ROOT / data["doc_path"]).read_text()
    headings = {
        _github_heading_slug(line.lstrip("#").strip())
        for line in doc_text.splitlines()
        if line.startswith("#")
    }
    for law in data["laws"]:
        assert law["doc_anchor"] in headings, f"missing anchor {law['doc_anchor']} in {data['doc_path']}"
        for sub in law["sub_laws"]:
            assert sub["doc_anchor"] in headings, f"missing sub-law anchor {sub['doc_anchor']} in {data['doc_path']}"


def test_laws_registry_supports_arbitrary_dynamic_law_ids(tmp_path):
    assert laws_registry.load_laws(tmp_path) is None
    assert laws_registry.top_level_laws(tmp_path) == []
    assert laws_registry.all_law_ids(tmp_path) == []

    _make_registry(tmp_path)
    summary = laws_registry.validate_laws_registry(tmp_path)
    assert summary == {
        "status": "valid",
        "schema_version": 1,
        "epistemic_status": "provisional_evolving",
        "laws": 2,
        "sub_laws": 1,
    }
    assert laws_registry.top_level_law_ids(tmp_path) == ["law_alpha", "law_beta"]
    assert laws_registry.all_law_ids(tmp_path) == ["law_alpha", "law_alpha_1", "law_beta"]
    assert [law["id"] for law in laws_registry.top_level_laws(tmp_path)] == ["law_alpha", "law_beta"]


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"schema_version": 2}, "schema_version"),
        ({"epistemic_status": "   "}, "epistemic_status"),
        ({"epigraph": "   "}, "epigraph"),
        ({"doc_path": "docs/missing.md"}, "existing repository file"),
        ({"doc_path": "../escape.md"}, "repository-relative path"),
        ({"laws": []}, "nonempty list"),
    ],
)
def test_laws_registry_rejects_malformed_top_level_fields(tmp_path, overrides, match):
    _make_registry(tmp_path, **overrides)
    with pytest.raises(ValueError, match=match):
        laws_registry.validate_laws_registry(tmp_path)


def test_laws_registry_rejects_duplicate_ids_and_invalid_status(tmp_path):
    payload = _make_registry(tmp_path)
    dup = read_json(tmp_path / laws_registry.LAWS_REGISTRY)
    dup["laws"][1]["id"] = "law_alpha"
    atomic_json(tmp_path / laws_registry.LAWS_REGISTRY, dup)
    with pytest.raises(ValueError, match="duplicate law id"):
        laws_registry.validate_laws_registry(tmp_path)

    dup_sub = read_json(tmp_path / laws_registry.LAWS_REGISTRY)
    dup_sub["laws"][1]["id"] = "law_beta"
    dup_sub["laws"][1]["sub_laws"] = [
        {"id": "law_alpha_1", "short_label": "Dup", "title": "Dup", "doc_anchor": "law-beta--dynamics"}
    ]
    atomic_json(tmp_path / laws_registry.LAWS_REGISTRY, dup_sub)
    with pytest.raises(ValueError, match="duplicate law id"):
        laws_registry.validate_laws_registry(tmp_path)

    bad_status = read_json(tmp_path / laws_registry.LAWS_REGISTRY)
    bad_status["laws"][1]["id"] = "law_beta"
    bad_status["laws"][1]["sub_laws"] = []
    bad_status["laws"][0]["status"] = "bogus"
    atomic_json(tmp_path / laws_registry.LAWS_REGISTRY, bad_status)
    with pytest.raises(ValueError, match="must be one of"):
        laws_registry.validate_laws_registry(tmp_path)
    assert payload["schema_version"] == 1
