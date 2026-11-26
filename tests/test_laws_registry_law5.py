"""Schema, anchor, provisional Law 5, mathematical control, and compatibility tests for Track C."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pytest

from experiments.forge import experiment_docs, family_documentation, hf_publish, laws_registry
from scripts.four_laws_audit.p9_verify_receipts import audit_four_laws_links


ROOT = Path(__file__).resolve().parents[1]
FOUR_LAWS_DOC = ROOT / "docs/four-laws.md"
AXIOM_FAMILY_DOC = ROOT / "configs/forge/family-documentation/axiom.json"


def _github_heading_slugs(markdown_text: str) -> set[str]:
    stripped = re.sub(r"```.*?```", "", markdown_text, flags=re.S)
    seen: dict[str, int] = {}
    anchors: set[str] = set()
    for heading in re.findall(r"^#{1,6} (.*)$", stripped, flags=re.M):
        slug = re.sub(r"[^\w\s-]", "", heading.strip().lower()).replace(" ", "-")
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(f"{slug}-{count}" if count else slug)
    return anchors


def test_laws_registry_loads_active_four_and_provisional_law5():
    summary = laws_registry.validate_laws_registry(ROOT)
    assert summary == {
        "status": "valid",
        "schema_version": 1,
        "epistemic_status": "provisional_evolving",
        "laws": 5,
        "sub_laws": 12,
    }

    registry = laws_registry.load_laws(ROOT)
    assert registry is not None
    assert laws_registry.top_level_law_ids(ROOT) == ["law1", "law2", "law3", "law4", "law5"]
    assert [law["id"] for law in laws_registry.active_laws(ROOT)] == ["law1", "law2", "law3", "law4"]
    assert [law["id"] for law in laws_registry.provisional_laws(ROOT)] == ["law5"]
    assert [law["id"] for law in laws_registry.laws_by_status(ROOT, ("active", "provisional"))] == [
        "law1", "law2", "law3", "law4", "law5"
    ]

    with pytest.raises(ValueError, match="must be one of"):
        laws_registry.laws_by_status(ROOT, "nonexistent_status")

    by_id = {law["id"]: law for law in registry["laws"]}
    law5 = by_id["law5"]
    assert law5["short_label"] == "Law 5"
    assert law5["title"] == "Observation & Statistical-Metric Identifiability"
    assert law5["status"] == "provisional"
    assert law5["doc_anchor"] == "law-5--observation--statistical-metric-identifiability"
    assert [sub["id"] for sub in law5["sub_laws"]] == ["law5a", "law5b", "law5c"]

    # Law 1A/1B sub-laws declared and Law 3B preserved for backward compatibility
    assert [sub["id"] for sub in by_id["law1"]["sub_laws"]] == ["law1a", "law1b"]
    assert [sub["id"] for sub in by_id["law3"]["sub_laws"]] == ["law3a", "law3b"]
    assert [sub["id"] for sub in by_id["law4"]["sub_laws"]] == [
        "law4_1", "law4_2", "law4_3", "law4_4", "law4_5"
    ]


def test_four_laws_doc_anchors_links_and_required_amendments():
    doc_text = FOUR_LAWS_DOC.read_text(encoding="utf-8")
    headings = _github_heading_slugs(doc_text)
    registry = laws_registry.load_laws(ROOT)
    assert registry is not None

    for law in registry["laws"]:
        assert law["doc_anchor"] in headings, f"missing anchor {law['doc_anchor']}"
        for sub in law["sub_laws"]:
            assert sub["doc_anchor"] in headings, f"missing sub-law anchor {sub['doc_anchor']}"

    # Verify every relative link and internal anchor in docs/four-laws.md resolves cleanly
    stripped = re.sub(r"```.*?```", "", doc_text, flags=re.S)
    link_re = re.compile(r"!?\[(?:[^\]\[]|\[[^\]]*\])*\]\(([^)\s]+)\)")
    anchor_cache: dict[Path, set[str]] = {FOUR_LAWS_DOC.resolve(): headings}
    broken: list[str] = []
    for target in link_re.findall(stripped):
        if re.match(r"^[a-z]+:", target):
            continue
        file_part, _, anchor = target.partition("#")
        dest = (FOUR_LAWS_DOC.parent / file_part).resolve() if file_part else FOUR_LAWS_DOC.resolve()
        if not dest.exists():
            broken.append(target)
            continue
        if anchor and dest.suffix == ".md" and not re.fullmatch(r"L\d+(?:-L\d+)?", anchor):
            if dest not in anchor_cache:
                anchor_cache[dest] = _github_heading_slugs(dest.read_text(encoding="utf-8"))
            if anchor not in anchor_cache[dest]:
                broken.append(f"{target} (missing anchor #{anchor})")
    assert not broken, f"Broken links/anchors in docs/four-laws.md: {broken}"

    # Verify P9 receipt-integrity link auditor passes cleanly on docs/four-laws.md + ledger
    audit = audit_four_laws_links(ROOT)
    assert audit["broken_links"] == []
    assert audit["missing_readouts"] == []

    # Verify concrete Track C amendments and Provisional Law 5 formalization content
    assert "### Law 5 — Observation & Statistical-Metric Identifiability" in doc_text
    assert "Law 1A" in doc_text and "U_1" in doc_text and "U_2" in doc_text
    assert "Law 1B" in doc_text and r"\rho_{\text{atom}}" in doc_text
    assert r"s_{\text{data}}" in doc_text and "Law 2 × 4.4" in doc_text
    assert "dim > 1024" in doc_text and "dim > 16" in doc_text
    assert "Cross-Operator" in doc_text and "pathology-witnessed" in doc_text.lower()
    assert r"\Delta y_i" in doc_text and "shock-triggered" in doc_text.lower()
    assert "Overarching Variational Meta-Objective" in doc_text or "overarching variational meta-objective" in doc_text.lower()
    assert "../reports/axiom/axiom-v4-failure-analysis.md" in doc_text
    # Ensure exact Law 5 population-vs-finite-batch distinction is preserved
    assert r"W_2^2(M_1 \odot X, M_2 \odot Y) = 0" in doc_text
    assert r"X^\star = p\,\mathbb{E}[Y]" in doc_text


def test_all_family_docs_and_downstream_loaders_accept_provisional_law5():
    fam_summary = family_documentation.validate_family_documentation(ROOT)
    assert fam_summary["status"] == "valid"
    assert fam_summary["families"] >= 40

    axiom_doc = family_documentation.read_json(AXIOM_FAMILY_DOC)
    assert "law5" in axiom_doc["law_mechanisms"]
    assert axiom_doc["law_mechanisms"]["law5"]["status"] == "partial"
    assert "reports/axiom/axiom-v4-failure-analysis.md" in axiom_doc["sources"]

    # Downstream callers load law5 with status="provisional" cleanly
    exp_reg = experiment_docs.load_laws_registry(ROOT)
    exp_law5 = next(law for law in exp_reg["laws"] if law["id"] == "law5")
    assert exp_law5["status"] == "provisional"
    by_id, parent_of = experiment_docs._law_lookup(exp_reg)
    assert parent_of["law5a"] == "law5"
    assert parent_of["law5b"] == "law5"
    assert parent_of["law5c"] == "law5"
    assert parent_of["law3b"] == "law3"
    assert "law1a" in by_id and parent_of["law1a"] == "law1"
    assert "law1b" in by_id and parent_of["law1b"] == "law1"

    laws_meta, laws_list = hf_publish._load_laws_registry_payload(
        ROOT, pages_base_url=hf_publish.DEFAULT_PAGES_BASE_URL
    )
    assert laws_meta["source"] == "configs/forge/laws.json"
    hf_law5 = next(law for law in laws_list if law["id"] == "law5")
    assert hf_law5["status"] == "provisional"
    assert hf_law5["doc_anchor"] == "law-5--observation--statistical-metric-identifiability"


def test_semantic_math_controls_for_law1a_sketch_bias_and_law5_independent_mask_shrinkage():
    """Numerical controls verifying the exact mathematical identities cited in Law 1A and Law 5."""
    # Control 1 (Law 1A): Single orthonormal sketch vs two-independent-sketch U-statistic at d_z=32, n_dirs=8
    rng = np.random.default_rng(0)
    d_z, n_dirs, v_target = 32, 8, 1.0
    n_trials = 400
    c_grid = np.linspace(0.05, 1.5, 60)

    # Pre-sample paired independent Haar-orthonormal sketches U1, U2 (d_z x n_dirs)
    u1_list, u2_list = [], []
    for _ in range(n_trials):
        q1, _ = np.linalg.qr(rng.standard_normal((d_z, n_dirs)))
        q2, _ = np.linalg.qr(rng.standard_normal((d_z, n_dirs)))
        u1_list.append(q1)
        u2_list.append(q2)

    single_losses = []
    ustat_losses = []
    target_mat = v_target * np.eye(d_z)
    scale = d_z / n_dirs
    for c in c_grid:
        # For isotropic Sigma = c * I, J_G = sqrt(c) * I so hat_Sigma_U = (d_z/n_dirs) * c * U U^T
        s_acc = 0.0
        u_acc = 0.0
        for q1, q2 in zip(u1_list, u2_list):
            sh1 = scale * c * (q1 @ q1.T)
            sh2 = scale * c * (q2 @ q2.T)
            d1 = sh1 - target_mat
            d2 = sh2 - target_mat
            s_acc += float(np.sum(d1 * d1))
            u_acc += float(np.sum(d1 * d2))
        single_losses.append(s_acc / n_trials)
        ustat_losses.append(u_acc / n_trials)

    c_star_single = c_grid[int(np.argmin(single_losses))]
    c_star_ustat = c_grid[int(np.argmin(ustat_losses))]
    assert c_star_single == pytest.approx(n_dirs / d_z * v_target, abs=0.03)
    assert c_star_ustat == pytest.approx(v_target, abs=0.05)

    # Verify exact deterministic identity on isotropic Sigma = c * I:
    # ||hat_Sigma_U - v I||_F^2 = ||Sigma - v I||_F^2 + ((d_z - n_dirs) / n_dirs) * tr(Sigma^2)
    c_test = 0.73
    sh = scale * c_test * (u1_list[0] @ u1_list[0].T)
    lhs = float(np.sum((sh - target_mat) ** 2))
    rhs = d_z * (c_test - v_target) ** 2 + ((d_z - n_dirs) / n_dirs) * (d_z * c_test ** 2)
    assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-12)

    # Control 2 (Law 5A): Independent-mask expected L2^2 identity & minimizer X* = p * E[Y]
    p = 0.10
    dim = 64
    y_vec = rng.standard_normal(dim)
    x_scales = np.linspace(0.0, 1.2, 49)
    n_mask = 4000
    m1 = (rng.random((n_mask, dim)) < p).astype(np.float64)
    m2 = (rng.random((n_mask, dim)) < p).astype(np.float64)

    emp_costs = []
    exact_costs = []
    for alpha in x_scales:
        x_vec = alpha * y_vec
        emp = float(np.mean(np.sum((m1 * x_vec - m2 * y_vec) ** 2, axis=1)))
        exact = float(p * np.sum((x_vec - p * y_vec) ** 2) + p * (1.0 - p ** 2) * np.sum(y_vec ** 2))
        emp_costs.append(emp)
        exact_costs.append(exact)
    assert np.allclose(emp_costs, exact_costs, rtol=0.05, atol=0.15)
    assert x_scales[int(np.argmin(exact_costs))] == pytest.approx(p, abs=1e-12)
    assert x_scales[int(np.argmin(emp_costs))] == pytest.approx(p, abs=0.03)
