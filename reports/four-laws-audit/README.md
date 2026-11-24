# ParticleGAN Four Laws — Mathematical Audit & Research Investigation

Integration branch: `research/four-laws` (branched from `origin/develop` at `0d18245e9`).

## Purpose

Adversarial scientific audit of the mathematical claims, experimental receipts, and theoretical framework in [`docs/four-laws.md`](../../docs/four-laws.md) and its associated implementations and Forge studies.

## Directory Layout (Ownership Boundaries)

- `reports/four-laws-audit/p0-law43-rank-audit.md` — P0: Law 4.3 exact rank vs effective rank & Newton–Schulz audit
- `reports/four-laws-audit/p1-law42-scale-settling.md` — P1: Law 4.2 scale invariance vs continuous settling impossibility
- `reports/four-laws-audit/p2-law3-identifiability.md` — P2: Law 3 non-identifiability proofs, factorial audit & analytic 1-Lipschitz controls
- `reports/four-laws-audit/p3-game-dynamics-counterexample.md` — P3: Joint adversarial game dynamics & 4-Laws counterexample hunt
- `reports/four-laws-audit/p4-function-space-generality.md` — P4: Function-space optimization generality beyond GANs
- `reports/four-laws-audit/p5-optimizer-asymmetry.md` — P5: Generator–critic optimizer asymmetry & adaptive pathology conditioning
- `reports/four-laws-audit/p6-law2-autodiff-audit.md` — P6: Law 2 forward-mode / mixed-derivative & computational regimes audit
- `reports/four-laws-audit/p7-law1-topology-audit.md` — P7: Law 1 measure/prior topology vs Wasserstein approximation & invariants
- `reports/four-laws-audit/p8-qualification-audit.md` — P8: Scientific qualification & Forge overfitting audit
- `reports/four-laws-audit/p9-receipts-efficiency-audit.md` — P9: Primary receipt verification & efficiency/Occam audit
- `reports/four-laws-audit/p10-synthesis-and-ledger.md` — P10: Unifying geometric synthesis, stronger formulations, evidence ledger & backlog
- `scripts/four_laws_audit/` — Reproducible mathematical & experimental scripts (one per track)
- `tests/four_laws_audit/` — Reproducible counterexamples and regression tests (one per track)

## Methodology Standard

Every investigated claim is classified as one of:
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

Bulk raw logs and checkpoints stay out of Git (per `AGENTS.md`); compact reports, final metrics, reproducible scripts, and regression tests are committed.
