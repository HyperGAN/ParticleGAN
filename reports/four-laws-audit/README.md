# ParticleGAN Four Laws — Scientific Audit, Canonical Formalization & Research Backlog

Integration branch: `research/four-laws` (branched from `origin/develop` at `0d18245e9`, reconciled with concurrent `origin/develop` PR #644 `36e287f98`).

## Purpose & Governance

Adversarial scientific audit, mathematical formalization, receipt verification, and research integration of ParticleGAN's **Four Laws of Adversarial Generative Learning**.

- **Open-Forge Principle:** Forge remains an **open, evolving research benchmark**. This audit does **not** freeze benchmark membership, thresholds, or seeds, does **not** remove difficult tests, does **not** modify historical receipts (`reports/forge/records/*.json`) or production `particlegan/` code, and does **not** declare any recipe scientifically qualified.
- **Theory–Implementation Separation:** Concrete ParticleGAN recipes (`unified_v11`, `unified_v12`, `unified_v13`, `minimal`) are treated strictly as replaceable candidate realizations, never as the laws themselves.

---

## Canonical Bedrock & Phase-2 Integration Documents (Start Here)

| Document | Scope & Audience |
| :--- | :--- |
| **[`docs/four-laws.md`](../../docs/four-laws.md)** | **Canonical Theoretical Bedrock:** Implementation-independent formalization of Laws 1–4 (each with the exact 9-subsection formal specification), Inter-Law Relationships, Energy-Optimal Learning (9 energy categories), Mechanism-to-Law separation, and verified current-implementation scope. |
| **[`integration-report.md`](integration-report.md)** | **Concluding Executive Integration Report (Section 10):** Crisp answers to all 8 core integration questions — what we know, what we were wrong about, what remains conjectural, what current code accomplishes (`v11`/`v12`/`v13`/`minimal`), where it fails, 10 falsifiable predictions, design principles for a new algorithm family, and merge readiness. |
| **[`evidence-ledger-and-revisions.md`](evidence-ledger-and-revisions.md)** | **Canonical Scientific Evidence Ledger, Claim Revision Record & Mechanism Catalog (Sections 6 & 7):** Traces every substantial theoretical & empirical claim to primary Forge receipts, P0–P10 tracks, tests, and prior literature under the 10-category taxonomy; Epistemic Limitations Matrix; side-by-side record of claims Corrected, Narrowed, Abandoned, or Introduced (`0d18245e9 -> 36e287f98`); and 5-part Mechanism-to-Law Separation Catalog (all 8 major mechanisms incl. PR #644's `feydy_atomic`). |
| **[`research-backlog-and-forge-proposals.md`](research-backlog-and-forge-proposals.md)** | **Prioritized Research Backlog, Code-Defect Inventory & Open-Forge Proposals (Sections 8 & 9):** Explicit Open-Forge banner; prioritized discriminating probes (`E1–E11`), open proof opportunities (`O1–O7`), and GPU studies (`G1–G6`); strictly decoupled `P0–P3` code-defect backlog (`D-01..D-13` with post-PR-#644 `file:line`); 10 proposed Open-Forge falsification tests; redundancy audit (zero hard-test removals); 10 sharper observables; and conceptual separation of open discovery vs. future post-freeze validation (`FUSQP-v1`). |
| **[`ci-and-consistency-check.md`](ci-and-consistency-check.md)** | **CI Parity, Concurrent `develop` (PR #644) Sync & Cross-Report Consistency Audit:** Reconciles PR #644 (`unified_v13` / ByteGAN), documents post-PR-#644 LOC and shifted line numbers, audits P0–P10 cross-report contradictions, and verifies CPU-only CI parity (`check_lint.py`, `check_repo_bloat.py`, `pytest`, `ci_pytest.py`, wheel/sdist exclusion). |

---

## Complete Phase-1 Audit Index (P0–P10 Reports, Reproducible Scripts & Automated Tests)

| Track | Report | Reproducible Script(s) | Automated Proof & Regression Tests | Core Verdict |
| :---: | :--- | :--- | :--- | :--- |
| **P0** | [`p0-law43-rank-audit.md`](p0-law43-rank-audit.md) | [`p0_rank_experiments.py`](../../scripts/four_laws_audit/p0_rank_experiments.py) | [`test_p0_law43_rank.py`](../../tests/four_laws_audit/test_p0_law43_rank.py) | Law 4.2 (ex-4.3) exact rank vs. stable/effective rank, Shampoo–polar equivalence, NS subspace invariance, BF16 rounding leakage & off-basis Hadamard distortion. |
| **P1** | [`p1-law42-scale-settling.md`](p1-law42-scale-settling.md) | [`p1_scale_settling.py`](../../scripts/four_laws_audit/p1_scale_settling.py) | [`test_p1_law42_scale_settling.py`](../../tests/four_laws_audit/test_p1_law42_scale_settling.py) | Law 4.4 (ex-4.2) Memoryless Impossibility Theorem, stateful trajectory degree-0 trade-offs, positive coherence floors, 6 code defects & 5-regime benchmark. |
| **P2** | [`p2-law3-identifiability.md`](p2-law3-identifiability.md) | [`p2_law3_analytic_lipschitz.py`](../../scripts/four_laws_audit/p2_law3_analytic_lipschitz.py) | [`test_p2_law3_identifiability.py`](../../tests/four_laws_audit/test_p2_law3_identifiability.py) | Law 3 split (3A vs. 3B), $W_1$ non-identifiability vs. $W_2$ Brenier, provably 1-Lip `G=Id` controls, generalized midpoint law & screening dead zone. |
| **P3** | [`p3-game-dynamics-counterexample.md`](p3-game-dynamics-counterexample.md) | [`p3_game_dynamics.py`](../../scripts/four_laws_audit/p3_game_dynamics.py) | [`test_p3_game_dynamics.py`](../../tests/four_laws_audit/test_p3_game_dynamics.py) | Four-Laws sufficiency falsified (`A1`, `A2`, `A3`, `B`), hidden $`\mu_D > 0`$ dissipation, 1:1 alternation $`\det(M_{\text{alt}}) = 1-\eta_D\mu_D`$, Kreiss transients, Schur–Cohn & OGDA $\lambda=0.5$. |
| **P4** | [`p4-function-space-generality.md`](p4-function-space-generality.md) | [`p4_funcspace_experiments.py`](../../scripts/four_laws_audit/p4_funcspace_experiments.py) | [`test_p4_funcspace_generality.py`](../../tests/four_laws_audit/test_p4_funcspace_generality.py) | Law 4.1/4.3 beyond-GAN generality of realized pushforward trust vs. JVP, TRPO, LARS, NG & Armijo; non-$`C^1`$ one-shot `_band_scale` & `b_p=2` probe blind spots. |
| **P5** | [`p5-optimizer-asymmetry.md`](p5-optimizer-asymmetry.md) | [`p5_asymmetry_experiments.py`](../../scripts/four_laws_audit/p5_asymmetry_experiments.py) | [`test_p5_asymmetry.py`](../../tests/four_laws_audit/test_p5_asymmetry.py) | Law 4.2 (ex-4.4) falsified as role-intrinsic (reverses on `3L G + 16L D`), Frobenius–Spectral Dilemma, `2.9×–7.8×` spatial force-field amplification, Law 2 one-sidedness & `Adaptive-Pathology` trade-off. |
| **P6** | [`p6-law2-autodiff-audit.md`](p6-law2-autodiff-audit.md) | [`p6_law2_autodiff.py`](../../scripts/four_laws_audit/p6_law2_autodiff.py) | [`test_p6_law2_autodiff.py`](../../tests/four_laws_audit/test_p6_law2_autodiff.py) | Law 2 RoF (`fwAD`) mixed 2nd-order derivatives, BF16 micro-secant collapse, opposite-signed Jensen bias, separable variance & RTX A6000 scaling crossover. |
| **P7** | [`p7-law1-topology-audit.md`](p7-law1-topology-audit.md) | [`p7_law1_topology.py`](../../scripts/four_laws_audit/p7_law1_topology.py) | [`test_p7_law1_topology.py`](../../tests/four_laws_audit/test_p7_law1_topology.py) | Law 1 exact support vs. sharp Borell–Sudakov–Tsirelson isoperimetric bounds, code invariant audit (`w_fake_ema`, LTV, Laguerre, `5×–6×` syncs) & zero-discriminator test. |
| **P8** | [`p8-qualification-audit.md`](p8-qualification-audit.md) | [`p9_verify_receipts.py`](../../scripts/four_laws_audit/p9_verify_receipts.py) | [`test_p9_receipt_integrity.py`](../../tests/four_laws_audit/test_p9_receipt_integrity.py) | Adaptive Forge overfitting (`68` -> `69` cards / `78.27` GPU-h, `v12` `0/2` unseen `ByteGAN`, L4 failures), calibration `BLOCKED` governance drift & future `FUSQP-v1` concept. |
| **P9** | [`p9-receipts-efficiency-audit.md`](p9-receipts-efficiency-audit.md) | [`p9_verify_receipts.py`](../../scripts/four_laws_audit/p9_verify_receipts.py) | [`test_p9_receipt_integrity.py`](../../tests/four_laws_audit/test_p9_receipt_integrity.py) | Primary receipt verification (3 broken links repaired), `r256` precision/harness confounds, idle A6000 benchmark & `8,564` -> `8,579`-LOC Occam audit. |
| **P10** | [`p10-synthesis-and-ledger.md`](p10-synthesis-and-ledger.md) | *(Phase-1 Master Synthesis of P0–P9)* | *(Full `tests/four_laws_audit/` suite)* | Unifying geometric evaluation (`Claim P10-C1`), all `83` Phase-1 8-point claims, `T1–T15` theorem catalog, `E1–E9` / `O1–O6` backlog & beyond-GAN assessment. |

---

## Epistemic Classification Standard (Canonical 10-Category Taxonomy)

Every claim in [`evidence-ledger-and-revisions.md`](evidence-ledger-and-revisions.md) carries strictly **one** primary label from the Section-6 taxonomy (mapped 1-to-1 to Phase-1 P0–P10 labels):
- `Proven mathematical result`
- `Necessary condition under explicit assumptions`
- `Sufficient condition under explicit assumptions`
- `Information-theoretic obstruction`
- `Empirically supported causal mechanism`
- `Empirical observation`
- `Engineering heuristic`
- `Conjecture`
- `Falsified claim`
- `Open question`

## Running the Complete Audit & CI Verification Suite

```bash
# Run all automated mathematical proofs, counterexamples, receipt controls & ByteGAN CPU tests (CPU-only CI parity)
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD" \
  /nvme/dev/hypergan/ParticleGAN/.venv/bin/pytest -v tests/four_laws_audit/ tests/test_unified_bytegan.py

# Check zero lint additions and zero repo bloat violations
PYTHONPATH="$PWD" /nvme/dev/hypergan/ParticleGAN/.venv/bin/python scripts/check_lint.py
PYTHONPATH="$PWD" /nvme/dev/hypergan/ParticleGAN/.venv/bin/python scripts/check_repo_bloat.py
```

Bulk raw logs and checkpoints stay out of Git (per [`AGENTS.md`](../../AGENTS.md)); compact reports, final metrics, reproducible scripts, and regression tests are committed. Historical Forge receipts (`reports/forge/records/*.json`) and production `particlegan/` modules are strictly untouched.
