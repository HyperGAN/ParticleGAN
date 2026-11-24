# ParticleGAN Four Laws — Mathematical Audit & Research Investigation

Integration branch: `research/four-laws` (branched from `origin/develop` at `0d18245e9`).

## Purpose

Adversarial scientific audit of the mathematical claims, experimental receipts, code invariants, and theoretical framework in [`docs/four-laws.md`](../../docs/four-laws.md) and its associated implementations and Forge studies.

**Start here:** **[`p10-synthesis-and-ledger.md`](p10-synthesis-and-ledger.md)** — Master synthesis evaluating the unifying geometric framework, stronger implementation-independent formulations of Laws 1–4, complete claim-by-claim evidence ledger across P0–P10, proven impossibility results & open proof opportunities, prioritized discriminating experiments & code-defect backlog, qualification roadmap (`FUSQP-v1`), and ruthless beyond-GAN contribution assessment.

## Complete Audit Index (Reports, Reproducible Scripts & Automated Tests)

| Track | Report | Reproducible Script(s) | Automated Proof & Regression Tests | Core Verdict |
| :---: | :--- | :--- | :--- | :--- |
| **P0** | [`p0-law43-rank-audit.md`](p0-law43-rank-audit.md) | [`p0_rank_experiments.py`](../../scripts/four_laws_audit/p0_rank_experiments.py) | [`test_p0_law43_rank.py`](../../tests/four_laws_audit/test_p0_law43_rank.py) | Law 4.3 exact rank vs. stable/effective rank, Shampoo–polar equivalence, NS subspace invariance, BF16 rounding leakage & off-basis Hadamard distortion. |
| **P1** | [`p1-law42-scale-settling.md`](p1-law42-scale-settling.md) | [`p1_scale_settling.py`](../../scripts/four_laws_audit/p1_scale_settling.py) | [`test_p1_law42_scale_settling.py`](../../tests/four_laws_audit/test_p1_law42_scale_settling.py) | Law 4.2 Memoryless Impossibility Theorem, stateful trajectory degree-0 trade-offs, positive coherence floors, 6 code defects & 5-regime benchmark. |
| **P2** | [`p2-law3-identifiability.md`](p2-law3-identifiability.md) | [`p2_law3_analytic_lipschitz.py`](../../scripts/four_laws_audit/p2_law3_analytic_lipschitz.py) | [`test_p2_law3_identifiability.py`](../../tests/four_laws_audit/test_p2_law3_identifiability.py) | Law 3 split (3A vs. 3B), $W_1$ non-identifiability vs. $W_2$ Brenier, provably 1-Lip `G=Id` controls, generalized midpoint law & screening dead zone. |
| **P3** | [`p3-game-dynamics-counterexample.md`](p3-game-dynamics-counterexample.md) | [`p3_game_dynamics.py`](../../scripts/four_laws_audit/p3_game_dynamics.py) | [`test_p3_game_dynamics.py`](../../tests/four_laws_audit/test_p3_game_dynamics.py) | Four-Laws sufficiency falsified (`A1`, `A2`, `A3`, `B`), hidden $\mu_D > 0$ dissipation, 1:1 alternation $\det(M_{\text{alt}}) = 1-\eta_D\mu_D$, Kreiss transients, Schur–Cohn & OGDA $\lambda=0.5$. |
| **P4** | [`p4-function-space-generality.md`](p4-function-space-generality.md) | [`p4_funcspace_experiments.py`](../../scripts/four_laws_audit/p4_funcspace_experiments.py) | [`test_p4_funcspace_generality.py`](../../tests/four_laws_audit/test_p4_funcspace_generality.py) | Beyond-GAN generality of realized pushforward trust vs. JVP, TRPO, LARS, NG & Armijo; non-$C^1$ one-shot `_band_scale` & `b_p=2` probe blind spots. |
| **P5** | [`p5-optimizer-asymmetry.md`](p5-optimizer-asymmetry.md) | [`p5_asymmetry_experiments.py`](../../scripts/four_laws_audit/p5_asymmetry_experiments.py) | [`test_p5_asymmetry.py`](../../tests/four_laws_audit/test_p5_asymmetry.py) | Law 4.4 falsified as role-intrinsic (reverses on `3L G + 16L D`), Frobenius–Spectral Dilemma, `2.9×–7.8×` spatial force-field amplification, Law 2 one-sidedness & `Adaptive-Pathology` trade-off. |
| **P6** | [`p6-law2-autodiff-audit.md`](p6-law2-autodiff-audit.md) | [`p6_law2_autodiff.py`](../../scripts/four_laws_audit/p6_law2_autodiff.py) | [`test_p6_law2_autodiff.py`](../../tests/four_laws_audit/test_p6_law2_autodiff.py) | Law 2 RoF (`fwAD`) mixed 2nd-order derivatives, BF16 micro-secant collapse, opposite-signed Jensen bias, separable variance & RTX A6000 scaling crossover. |
| **P7** | [`p7-law1-topology-audit.md`](p7-law1-topology-audit.md) | [`p7_law1_topology.py`](../../scripts/four_laws_audit/p7_law1_topology.py) | [`test_p7_law1_topology.py`](../../tests/four_laws_audit/test_p7_law1_topology.py) | Law 1 exact support vs. sharp Borell–Sudakov–Tsirelson isoperimetric bounds, code invariant audit (`w_fake_ema`, LTV, Laguerre, `5×–6×` syncs) & zero-discriminator test. |
| **P8** | [`p8-qualification-audit.md`](p8-qualification-audit.md) | [`p9_verify_receipts.py`](../../scripts/four_laws_audit/p9_verify_receipts.py) | [`test_p9_receipt_integrity.py`](../../tests/four_laws_audit/test_p9_receipt_integrity.py) | Adaptive Forge overfitting (`68` cards / `78.27` GPU-h, `0/2` unseen `ByteGAN`, L4 failures), calibration `BLOCKED` governance drift & `FUSQP-v1`. |
| **P9** | [`p9-receipts-efficiency-audit.md`](p9-receipts-efficiency-audit.md) | [`p9_verify_receipts.py`](../../scripts/four_laws_audit/p9_verify_receipts.py) | [`test_p9_receipt_integrity.py`](../../tests/four_laws_audit/test_p9_receipt_integrity.py) | Primary receipt verification (3 broken links repaired in `docs/four-laws.md`), `r256` precision/harness confounds, idle A6000 benchmark & `8,564`-LOC Occam audit. |
| **P10** | [`p10-synthesis-and-ledger.md`](p10-synthesis-and-ledger.md) | *(Synthesizes P0–P9 + repairs `docs/four-laws.md`)* | *(Full `tests/four_laws_audit/` suite)* | Unifying geometric evaluation, airtight Laws 1–4, master evidence ledger, impossibility catalog, prioritized defect/research backlog & beyond-GAN assessment. |

## Methodology Standard

Every investigated claim is classified under one primary label from the 10-category taxonomy and evaluated against the 8-point structured rubric:
- Proven mathematical result
- Mathematical conjecture
- Necessary condition under stated assumptions
- Sufficient condition under stated assumptions
- Information-theoretic obstruction
- Empirically supported causal mechanism
- Empirical correlation
- Engineering heuristic
- Falsified claim
- Unresolved question

## Running the Complete Audit Verification Suite

```bash
# Run all automated mathematical proofs, counterexamples, receipt & semantic controls
PYTHONPATH="$PWD" /nvme/dev/hypergan/ParticleGAN/.venv/bin/pytest -v tests/four_laws_audit/

# Check zero lint additions
PYTHONPATH="$PWD" /nvme/dev/hypergan/ParticleGAN/.venv/bin/python scripts/check_lint.py
```

Bulk raw logs and checkpoints stay out of Git (per `AGENTS.md`); compact reports, final metrics, reproducible scripts, and regression tests are committed. Historical Forge receipts (`reports/forge/records/*.json`) and production `particlegan/` modules are strictly untouched.
