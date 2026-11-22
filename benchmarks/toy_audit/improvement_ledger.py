"""Join the immutable toy audit and separately versioned follow-up evidence.

This launches no training. Scores describe the explicitly stated test question;
neither a new score nor a software PASS changes a trained qualification receipt.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


REVIEWS = {
    "source-family-11": (4, "Paired sign correspondence unit", "Full-budget paired and rejected-objective controls pass; the complete joint-reflection ambiguity has an exact witness. The task remains a scalar gain around a supplied expert."),
    "source-family-12": (4, "Safe-fast scalar landing objective unit", "Full-budget objective controls and the all-start denominator pass. Cost-only is faster, so this tests objective composition rather than adversarial superiority."),
    "develop-img_mean_discriminator": (4, "Mean-only critic nonidentifiability unit", "Reclassified as an analytic negative unit: every target template has the same mean, so the critic cannot identify the desired spatial law. Expected learned failure is not a positive qualification."),
    "develop-img_uniform_generator": (4, "Spatially uniform generator impossibility unit", "Reclassified as an analytic negative unit: best attainable stripe RMSE is sqrt(3/16)=.433, above the .10 quality neighborhood. Expected learned failure is not a positive qualification."),
    "develop-two_pole": (3, "Balanced two-pole finite-grid fidelity", "The new contract matches the actual 12-atom target, masses and quantile locations. The recorded mean bounds in-support rows at 6/12, so the old travel PASS does not qualify density. No initialization impossibility is claimed."),
    "source-family-15": (3, "Five fixed words and paired reconstruction", "The new contract includes confidence, uniform word mass, every padding token and correctly paired reconstruction. The vocabulary stays fixed; training outcomes remain separate."),
    "source-family-16": (3, "Single Gaussian distribution plumbing", "The new contract rejects mean-only, wrong-width and equal-covariance circle witnesses using moments and CDFs. It remains a unimodal plumbing example."),
}

DEFINITIONS = {
    "develop-two_pole": "Actual 12-atom grid, balanced mass and sorted quantile/support fidelity; retained travel metrics prove support failure without retraining.",
    "source-family-15": "Full six-token probability/confidence, word mass and exact paired reconstruction, including padding.",
    "source-family-16": "Mean, covariance eigenvalues, radial CDF and 16 fixed projected Gaussian CDFs.",
    "source-family-11": "Full joint-reflection witness and held-out paired action error against a nonzero neutral baseline.",
    "source-family-12": "Count all starts, crashes and timeouts; use restricted mean time instead of successful-start selection.",
    "source-family-13": "Held-out previous-command/action correspondence independent of action marginals; preserve live/EMA and the original accuracy ceiling.",
    "source-family-10": "Eight masses, within-mode covariance/radial law, shift-aware target and terminal stability; bind actual prior resources.",
    "source-family-14": "Separate paired, support, moving and replay contracts; require paired fidelity and positive code-removal loss under every judge.",
}

QUESTIONS = {
    "pr45-adapted": "Test an isotropic unit change with the published fixed setup versus jointly rescaled kernel lengths and prior initialization. This composite contrast does not establish that each rescaling is independently necessary.",
    "develop-img_mean_discriminator": "Verify that a mean-only critic cannot distinguish equal-mean spatial templates; retain this as an expected-failure negative control.",
    "develop-img_uniform_generator": "Verify the exact best-constant approximation bound for nonuniform stripes; retain an impossible host as an expected-failure negative control.",
    "develop-two_pole": "Recover the actual balanced 12-atom target grid around both poles; travel alone does not verify its mass, support or width.",
    "source-family-15": "Recover the uniform five-word law and the correct reconstruction for every canonical word, with confident probabilities and all padding tokens.",
    "source-family-16": "Recover N((1,1), .04 I), including its radial and projected laws; verify unimodal training/sample plumbing.",
}

FRESH_FILES = (
    ("source_families/coverage.json", "source_families/README.md", "records"),
    ("source-family-training.json", "SOURCE_FAMILY_TRAINING.md", "fixtures"),
    ("source-demos.json", "SOURCE_DEMOS.md", "cases"),
    ("conditional_sources/coverage.json", "conditional_sources/README.md", "records"),
    ("route_sources/coverage.json", "route_sources/README.md", "records"),
    ("paired_sources/coverage.json", "paired_sources/README.md", "records"),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(root, name, bindings):
    path = root / name
    bindings[name] = sha(path)
    return json.loads(path.read_text())


def fresh_records(root, bindings):
    grouped = defaultdict(list)
    for name, report, key in FRESH_FILES:
        if not (root / name).exists():
            continue
        data = read(root, name, bindings)
        for record in data[key]:
            cid = record["catalog_id"]
            media = record.get("gif_relative_to_report", record.get("media_path", record.get("media")))
            if isinstance(media, dict):
                media = media.get("path")
            if media:
                media = str(Path(name).parent / media)
            grouped[cid].append(dict(
                label=record.get("fixture", record.get("family", record.get("name", record.get("problem", cid)))),
                execution=record.get("fresh_execution_status", record.get("execution_status", record.get("status", "see receipt"))),
                original_scientific_status=record["original_scientific_status"],
                added_gate_status=record["added_gate_status"],
                media=media, report=report, receipt=name,
            ))
    return grouped


def build(root, *, require_fresh=False):
    bindings = {}
    catalog = read(root, "catalog.json", bindings)
    image = read(root, "image-quality-v2.json", bindings)
    vector = read(root, "vector-quality-controls.json", bindings)
    definition = read(root, "definition-quality-controls.json", bindings)
    diagnosis = read(root, "failure-diagnosis.json", bindings)
    readiness = read(root, "merge_readiness/merge-readiness.json", bindings)
    integration = read(root, "merge_readiness/local-integration.json", bindings)
    images = {row["id"]: row for row in image["cases"]}
    vectors = {row["id"]: row for row in vector["cases"]}
    originals = {row["id"]: row for row in catalog["cases"]}
    poor = {cid for cid, row in originals.items() if row["rating"] <= 3}
    image_poor = poor & images.keys()
    vector_poor = poor & vectors.keys()
    controls = set(definition["controls"])
    assert len(originals) == 109 and len(poor) == 64
    assert (len(image_poor), len(vector_poor), len(controls)) == (34, 22, 8)
    assert image_poor | vector_poor | controls == poor
    assert not (image_poor & vector_poor or image_poor & controls or vector_poor & controls)
    by_failure = defaultdict(list)
    for item in diagnosis["diagnoses"]:
        assert item["catalog_id"] in originals
        by_failure[item["catalog_id"]].append({key: item.get(key) for key in (
            "arm", "failure_class", "confidence", "established", "failed_final_gates",
            "unresolved_optimization_mechanism", "next_action")})
    fresh = fresh_records(root, bindings)
    if require_fresh:
        assert set(fresh) == {f"source-family-{number:02}" for number in range(17)}
    rows = []
    for cid, original in originals.items():
        review = REVIEWS.get(cid)
        rating, name = review[:2] if review else (original["rating"], original["name"])
        improvement, results, added_failures = None, [], []
        if cid in images:
            item = images[cid]
            improvement = item["improvement"]["implemented"]
            results.append(dict(cohort=image["protocol"], status=" / ".join(a["revised"]["status"] for a in item["arms"]), report="IMAGE_QUALITY_V2.md"))
            for arm in item["arms"]:
                if not arm["revised"]["passed"]:
                    added_failures.append(dict(arm=arm["label"], failed_terms=arm["failure_components"], structural_limit=arm["host_limitation"]))
        elif cid in vectors:
            item = vectors[cid]
            improvement = item["improvement"]
            results.append(dict(cohort=vector["version"], status=" / ".join(a["revised_status"] for a in item["arms"]), report="NON_IMAGE_QUALITY.md"))
            for arm in item["arms"]:
                if arm["revised_status"] == "FAIL":
                    added_failures.append(dict(arm=arm["arm"], final=arm["final"], terminal=arm["convergence"], remaining_cause="CDF mismatch identifies a distribution error, not its unique optimizer cause."))
        elif cid in controls:
            improvement = DEFINITIONS[cid]
            results.append(dict(cohort=definition["version"], status="Evaluator controls only; trained result separate", report="NON_IMAGE_QUALITY.md"))
            if cid == "develop-two_pole":
                results.append(dict(cohort="retained_mean_support_bound", status="Density support FAIL: at most 6/12 rows can be in support", report="NON_IMAGE_QUALITY.md"))
        if cid == "pr227":
            improvement = "Three gate paths now require a positive code-removal loss; four harmful-judge controls reject and saved endpoint deltas remain positive."
            results.append(dict(cohort="beneficial_signed_v2", status="Saved endpoint assertions PASS; follow-up training not rerun", report="merge_readiness/README.md"))
        if cid in ("develop-img_mean_discriminator", "develop-img_uniform_generator"):
            improvement += " Reclassified as an analytic negative unit; proof/control PASS is distinct from expected training FAIL."
            results.append(dict(cohort="analytic_negative_unit", status="Proof and negative software controls PASS", report="IMAGE_QUALITY_V2.md"))
        failures = by_failure[cid]
        if cid in ("pr224", "pr226", "pr227"):
            merge = "Candidate diagnostic; signed follow-up required for 227; fresh remote head/checks unavailable"
        elif original.get("pr") in (22, 45, 153, 196):
            merge = "HOLD: current API, frozen model gate or conflicts unresolved"
        elif original.get("pr"):
            merge = "Below authorized original top-tier cutoff; no merge proposed"
        else:
            merge = "Existing develop/reference definition; no new toy PR to merge"
        verifies = QUESTIONS.get(cid, original["verifies"])
        if cid in vectors and original.get("pr"):
            verifies = (
                f"Test distribution fidelity on {original['name']} under the published setup "
                "and its composite critic-architecture/prior-initialization control. "
                "The source lengthscale hypothesis is not isolated by this contrast; "
                "initialization is not matched between arms."
            )
        rows.append(dict(
            id=cid, original_name=original["name"], name=name, pr=original.get("pr"),
            original_rating=original["rating"], followup_rating=rating,
            quality_tier="Good bounded question" if rating >= 4 else "Well defined but narrow" if rating == 3 else "Weak application claim",
            rating_reason=review[2] if review else "Original scientific-quality score retained; added measurements do not imply convergence or independent causal identification.",
            verifies=verifies, original_verifies=original["verifies"],
            original_status=original["status"], original_media=original.get("media"),
            poor_definition_improvement=cid in poor, implemented=improvement,
            added_assessments=results, fresh_source_assessments=fresh.get(cid, []),
            original_failures=failures, added_failures=added_failures, merge_decision=merge,
        ))
    rows.sort(key=lambda row: (-row["followup_rating"], row["name"].casefold(), row["id"]))
    old_scores = Counter(row["original_rating"] for row in rows)
    new_scores = Counter(row["followup_rating"] for row in rows)
    media_cases = sum(bool(row["original_media"] or any(a["media"] for a in row["fresh_source_assessments"])) for row in rows)
    all_media = {row["original_media"] for row in rows if row["original_media"]}
    all_media.update(a["media"] for row in rows for a in row["fresh_source_assessments"] if a["media"])
    missing_media = [row["id"] for row in rows if not row["original_media"] and not any(a["media"] for a in row["fresh_source_assessments"])]
    for media in all_media:
        if not (root / media).is_file():
            raise ValueError(f"missing declared training visualization: {media}")
    return dict(
        schema="toy-improvement-ledger-v1", scope="All 109 reviewed entries; separate original and follow-up definitions/results; no production repairs",
        original_evidence_unchanged=True, source_bindings=bindings,
        generator_sha256=sha(Path(__file__)), merge_cutoff="Original quality tiers 5/5 and 4/5, plus a passing declared diagnostic and current remote checks",
        counts=dict(entries=len(rows), improved_poor_definitions=len(poor), image_poor=len(image_poor), vector_poor=len(vector_poor), other_poor=len(controls),
                    original_score_counts=dict(sorted(old_scores.items(), reverse=True)), followup_score_counts=dict(sorted(new_scores.items(), reverse=True)),
                    original_mean=sum(k*v for k,v in old_scores.items())/len(rows), followup_mean=sum(k*v for k,v in new_scores.items())/len(rows),
                    reassessed_definitions=len(REVIEWS), original_deficient_cases=diagnosis["counts"]["diagnosed_cases"], original_deficient_arms=diagnosis["counts"]["diagnosed_arms"],
                    entries_with_actual_training_media=media_cases, actual_training_gifs=len(all_media), fresh_source_families=len(fresh), actual_remote_merges=integration["actual_remote_merges"]),
        media_coverage=dict(missing_catalog_ids=missing_media, actual_gifs_sha256={name: sha(root/name) for name in sorted(all_media)},
                            claim="Only real observed states; partial and failed runs remain labelled; missing media are not inferred."),
        merge_local_suite=readiness["local_integrated_suite"], rows=rows,
    )


def cell(value):
    if isinstance(value, dict):
        value = "; ".join(f"{key} {item}" for key, item in value.items())
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown(data):
    counts = data["counts"]
    lines = ["# Improved definitions and sorted toy results", "",
        f"All **{counts['entries']} entries** remain in the denominator. All **{counts['improved_poor_definitions']} original entries rated 2/5 or 3/5** now have an implemented definition, evaluator or comparison improvement: 34 image, 22 vector and eight other definitions. This improves tests; it does not claim that their learned models were repaired.", "",
        "The separate follow-up definition scores are **7 × 5/5, 42 × 4/5, 57 × 3/5 and 3 × 2/5**, mean **3.49/5** (original 3.40/5). Seven scoped questions are reassessed below. Scores measure scientific usefulness; variant counts and this mean are not independent-family weights or model pass rates. Application-named colorization/inpainting/translation claims remain 2/5 because their generators receive no conditional query.", "",
        "The sorted tiers are **good bounded questions** at 4–5, **well defined but narrow** at 3, and **weak application claims** at 2. A scientifically good test can expose a failed model, while a narrow template can have a passing model.", "",
        "Original [ratings/results](PROBLEMS.md) and [catalog](catalog.json) are unchanged. New CDF/mass/paired gates are separately versioned; a stricter FAIL does not rewrite an old PASS. Source cohorts keep default, changed-resource and interrupted attempts separate.", "",
        f"**Training visualization coverage:** {counts['actual_training_gifs']} actual-state GIFs cover {counts['entries_with_actual_training_media']}/109 entries. {'No missing entries.' if not data['media_coverage']['missing_catalog_ids'] else 'Missing GIFs: ' + ', '.join('`'+cid+'`' for cid in data['media_coverage']['missing_catalog_ids']) + '. Their source/API/runtime blockers are explicit; no convergence is inferred.'} Partial GIFs show actual progress through the last retained state, not a completed-budget pass.", "",
        "**Merge status:** PR224 and PR226 are locally ready diagnostic candidates; PR227 has a committed signed-gate repair. Their prospective combined Git tree matches the already-tested overlay: **54 passed, one opt-in training test skipped, one strict XFAIL**. **Zero remote PRs merged:** authenticated GitHub access is blocked, so current heads/checks are unknown. [Exact readiness and holds](merge_readiness/README.md).", "",
        "## Definition score changes", "",
        "| Entry | Original → follow-up | What earned the scoped reassessment |", "|---|---:|---|"]
    for row in data["rows"]:
        if row["id"] in REVIEWS:
            lines.append(f"| `{row['id']}` · {cell(row['name'])} | {row['original_rating']} → {row['followup_rating']} | {cell(row['rating_reason'])} |")
    lines += ["", "## All entries, sorted by follow-up definition quality", "",
        "The result column always retains the original trained/diagnostic verdict. Added results and fresh GIFs are identified separately. Source-only, incomplete, ungated and blocked entries receive no inferred model PASS.", "",
        "| Score (original) | Problem and exact question | Original; added/fresh evidence | Improvement or remaining work | Media |", "|---:|---|---|---|---|"]
    for row in data["rows"]:
        name = cell(row["name"])
        if row["pr"]:
            name += f" · [PR{row['pr']}](https://github.com/HyperGAN/ParticleGAN/pull/{row['pr']})"
        evidence = [cell(row["original_status"])]
        evidence += [f"[Added]({a['report']}): {cell(a['status'])}" for a in row["added_assessments"]]
        evidence += [f"[{cell(a['label'])}]({a['report']}): execution {cell(a['execution'])}; original {cell(a['original_scientific_status'])}; added {cell(a['added_gate_status'])}" for a in row["fresh_source_assessments"]]
        action = row["implemented"] or (row["original_failures"][0]["next_action"] if row["original_failures"] else "Retain the declared bounded question and source-specific evidence.")
        media = [f"[Original GIF]({row['original_media']})"] if row["original_media"] else []
        media += [f"[{cell(a['label'])} GIF]({a['media']})" for a in row["fresh_source_assessments"] if a["media"]]
        lines.append(f"| {row['followup_rating']}/5 ({row['original_rating']}) | {name}<br>{cell(row['verifies'])} | {'<br>'.join(evidence)} | {cell(action)} | {'<br>'.join(media) or 'Missing; source-only or blocked'} |")
    lines += ["", "## Failure explanations and remaining causes", "",
        "[Failure diagnosis](FAILURE_DIAGNOSIS.md) covers every original failed arm and the ungated conditional deficiencies: **69 entries / 73 arms**. It binds failed bars, retained tensors, source hashes and confidence separately. Structural impossibility and the constructed controller cause are proved within their fixtures. Mass, shape, pairing and late-window errors identify what failed; most do not uniquely identify an optimizer cause.", "",
        "The [image revision](IMAGE_QUALITY_V2.md) rejects seven former terminal passes; the [vector revision](NON_IMAGE_QUALITY.md) rejects all 34 retained arms while independent target draws and declared finite-cloud witnesses pass. All 12 poor vector PR contrasts change prior initialization as well as critic components. Stronger scoring therefore does not establish an isolated critic remedy.", "",
        "Fresh source readouts explain the paired/sign/safe-fast successes, native accuracy shortfall, ring acquisition/resource mismatch, routed replay failure, capped support study and demo fidelity results. The remaining shipped families retain exact-source attempts, numeric checkpoint evidence or prerequisite blockers. Follow their linked reports for original budgets, laws, costs and actual frames. Configuration repairs remain the separate workstream.", "",
        "Rebuild this report without training:", "", "```sh", "python -m benchmarks.toy_audit.improvement_ledger --require-fresh", "```", "",
        "[Machine-readable ledger](improvements.json) binds all input receipts and records each original/new score, assessment, failed arm and next action.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-fresh", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2] / "reports/toy_audit"
    result = build(root, require_fresh=args.require_fresh)
    (root / "improvements.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (root / "IMPROVEMENTS.md").write_text(markdown(result))
    print(json.dumps(result["counts"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
