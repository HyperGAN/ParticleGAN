# CUDA inventory v2 interruption

The v2 inventory stopped because the generic vector adapter used the recipe's
400-update schedule horizon as ring16's execution limit. The task declared
1,600 updates. Both ring attempts raised `RuntimeError: recipe training budget
exhausted`; their original grades remain **INCOMPLETE**, with raw status `error`.
They supply no numerical ring failure verdict.

The executed source was
[`f802fcf9b04d32d2027321af0894a78a2d1343dd`](https://github.com/255BITS/ParticleGAN/commit/f802fcf9b04d32d2027321af0894a78a2d1343dd),
scientific source digest
`9255e2574da332fba5360b65868ad77824ccad17b5e3c994464420ecb187cad0`.
All active workers finished before **21 pending requests were withdrawn**.
The final queue contains 18 terminal and 626 unexecuted jobs, with two blocked
and 21 cancelled submissions. Reserved cost is zero; all 18 original charges
remain counted: **811.4173847100465 seconds**.

The original task results contain 13 PASS, three numerical FAIL and two
INCOMPLETE errors. These counts include two optional clock-audit PASS results.
The genuine numerical failures are two completed 20,001-update word tasks and
one completed two-pole task. The software interruption does not change them.
No candidate obtained the complete six-task Tier 1 barrier, and no Tier 2
qualification is claimed. Exact candidate IDs, grades, final metrics,
confirmation observations, costs and original receipt hashes are retained in
[the compact attempt readout](attempts.json).

The three measured Gaussian smoke cells passed. Their first independently
confirmed states were update 792 for the selected Ada-NSGDA configuration,
542 for the selected discriminator-only DualNorm configuration and 917 for the
selected DualNorm configuration. Each completed all 1,000 updates and all 24
paired observations. For the discriminator-only case at update 542, primary
and confirmation KS were 0.027592 and 0.028274, mean errors were 0.002316 and
0.002853 target standard deviations, and standard-deviation ratios were
0.929425 and 0.931786; both draws contained 4,096 finite samples.
Its terminal KS was 0.073502, so this acquisition success makes no retention
claim. [The actual-training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/gaussian-smoke-inventory/v2-interruption/media/gaussian1d_smoke.gif) uses only the saved
scored observations from that original v2 attempt; its
[render receipt](media/gaussian1d_smoke.json) records zero additional updates or
sampling draws. Numerical gates, rather than the animation, establish PASS.

The two ring error attempts were `6fa75d590a42437fa095c3e547a7b4b0`
(selected Ada-NSGDA) and `5206c70d0f1540c89801b52dfc7dfbdd`
(selected discriminator-only DualNorm). Each raw log retains 24 observations
ending at update 400. Neither saved a checkpoint, so a continuation from the
failed ring state is unavailable. Their charged costs are 7.535110180964693
and 8.487665095017292 seconds.

The v3 software amendment passes `task.execution.steps` explicitly to the
public trainer in vector and image adapters, while preserving
`original_schedule_horizon` in the recipe. Ring16 is the only current task
affected by the premature limit; the image change prevents the same omission
for future image tasks with distinct horizons. Gaussian, native continuation,
ring endurance, adaptation, clock audit and word hosts already own explicit
execution allowances. The full fixed roster runs under the repaired source
[`a4caa21d1039684d23e57be8e133b51b3b120775`](https://github.com/255BITS/ParticleGAN/commit/a4caa21d1039684d23e57be8e133b51b3b120775)
to establish one compatible qualification cohort. Existing v2 passes remain
under their actual source; they cannot be spliced into v3 qualification.
Recipes, seed 0, task budgets and roster remain fixed. Scientific-failure
retries are zero, and v2's paid cost remains part of the goal accounting.

Continue the registered v3 run and use its complete numerical results for
whole-candidate selection and Tier 2 eligibility. The v2 Gaussian passes
support the acquisition question; retention remains a separate Tier 2 claim.
The single goal leaderboard remains
[technique-inventory](../../technique-inventory.md).

The [archive receipt](provenance.json) binds a byte-exact local archive of the
entire stopped v2 queue, its frozen source snapshot, all 18 durable original
request/evidence/result directories, all 12 registered search reports, and the
round/campaign declarations. The archive is
`artifacts/gaussian-smoke-inventory-v2-interruption.tar.gz` (21,781,946 bytes),
SHA-256 `220b4310532c73a483827310201d11852089430d34079d03f93d33d7dfb5d488`.
Its embedded inventory certifies 1,527 original files (211,841,446 bytes);
every archived file was verified against its original size and SHA-256.
Raw logs, per-update streams, checkpoints and the full inventory remain outside
Git. This reports-only projection grants no qualification credit.

Verify a hydrated archive and restore the exact original relative paths into
an isolated directory:

```sh
/usr/bin/python reports/forge/gaussian-smoke-inventory/v2-interruption/archive.py \
  --archive artifacts/gaussian-smoke-inventory-v2-interruption.tar.gz --verify-only
mkdir -p /tmp/particlegan-inventory-v2-originals
tar -xzf artifacts/gaussian-smoke-inventory-v2-interruption.tar.gz \
  -C /tmp/particlegan-inventory-v2-originals
```

[The archive script](archive.py) reads saved files only. It can recreate the
archive and compact projections with `--source-root` pointing at a hydrated
original tree and `--archive` pointing at a new immutable archive path. It
imports no model, launches no worker and performs no training or sampling.
