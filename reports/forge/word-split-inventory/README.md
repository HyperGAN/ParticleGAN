# Tier 1 rerun after the five-word acquisition/hold split

All five runnable selected families pass five-word acquisition and finish at
**5/6 required Tier 1 passes**. BCAP's remaining failure is the scalar Gaussian;
K3P, KA2, R1/R2 and the released GAN v3 MoG recipe still fail ring16.
No family satisfies all six prerequisites, so **no ordinary Tier 2 run is
eligible**. E22 and Atlas remain blocked before reservation because these clean
tasks lack their policy-aware control and served-sampling contract.

This is the completed ordinary `discriminator_stability` revision-8 campaign,
executed from develop commit `737592c128ef84cf596c7f198b0ce8aad7c65700` on two
RTX A6000 GPUs. All **35 attempts** used CUDA: six required smoke tasks and the
optional clock audit for each runnable family. Paid worker cost was
**1,796.544469 seconds**; reservations are released. There were no scientific
retries, seed changes or hyperparameter tuning. Each family used its exact
previously selected whole configuration. Smoothing stayed off; the existing
DualNorm truncation and disabled autograd multithreading remained fixed.

Use the single [current technique inventory](../technique-inventory.md) for the
leaderboard. The [compact readout](readout.json) preserves every final metric,
gate, candidate/source identity, certificate projection, cost and prior-cohort
comparison. The [protocol](protocol.json) records the fixed comparison and
reservation contract. Tier 2's 21 required cells remain unmeasured; a prerequisite
failure is not a measured failure on those problems.

BCAP independently confirms the full five-word generation and paired inverse
goal first at **update 1,667**, with **13/24** confirmed checks and a passing
endpoint. K3P and KA2 first pass at update 834, with 16/24 and 17/24 checks.
R1/R2 first passes at 2,501 and the released GAN v3 recipe at 834; each has
only one confirmed passing check. These four recipes fail at the endpoint.
All runs complete 20,001 updates. Acquisition succeeds across these families,
while continuing to hold the solution remains a distinct question. The
[earlier BCAP hold diagnostic](../five-word-tier-split/README.md) failed 5 of
25 paired checks; it retains its task-only identity and supplies no ordinary
Tier 2 qualification here.

BCAP's Gaussian has one primary passing check at **update 167**. Its KS distance
is **0.047977** against the unchanged **0.05** limit; the independent same-state
confirmation is **0.054185** and fails. The state hashes match and confirmation
does not change training. At update 1,000, primary KS is **0.064603** and
confirmation KS is **0.064796**. Mean and standard deviation meet their bounds,
but the full distribution gate does not. These measurements do not establish
whether population KS is above or below the boundary; an isolated primary hit
does not certify this smoke test.

BCAP's Gaussian and ring16 endpoint scalar metrics match the
[previous post-truncation cohort](../gaussian-smoke-inventory/final-v6/readout.json)
exactly. Ring16 still passes all full-quality gates after 1,600 updates: 16 modes,
quality fraction 0.9560547, mass TV 0.0786133, component covariance error 0.4706781
and minimum component eigenvalue ratio 0.3873404. The present Gaussian failure
already occurred in that cohort. The word split changes the acquisition
qualification policy; it does not demonstrate a new optimizer regression or
confer retention credit. Endpoint equality is not a proof that every intermediate
state is equal; the readout labels its comparison scope explicitly.
All 31 available task comparisons across the five selected families have equal
scalar endpoint summaries, including the word tasks under their changed policy.

Keep acquisition in Tier 1 and strict own-checkpoint hold in Tier 2. The next
bounded BCAP comparison should use the already prepared
[positive smoothing search](../../../configs/forge/searches/bcap-dualnorm-smoothing-tier1-v1.json)
with the same winning global rates and unchanged smoke gates. Its constant
parameter makes small normalized updates sensitive to gradient magnitude and
does not introduce learning-rate annealing. A whole six-task pass would then
permit the ordinary Tier 2 stability tests. This publication does not execute
that search or adopt a new default.

Seven [actual-training GIFs and their receipts](media/index.json) were exported
from the selected BCAP attempts, including the failed Gaussian. Selection did
not depend on the grade. Export added zero optimizer updates and zero sampling
draws; numerical evidence determines the verdicts. Bulk checkpoints, arrays,
stdout and event streams live in the byte-verified archive named and hashed by
the readout, outside Git.
The [word smoke](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/word-split-inventory/media/five_word_joint_smoke.gif),
[Gaussian smoke](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/word-split-inventory/media/gaussian1d_smoke.gif) and
[ring16 acquisition](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/word-split-inventory/media/ring16_acquisition.gif) illustrate those measured goals.

The generic inventory launcher refused saved-configuration admission before
training because it did not attach a bounded registration. The reproduction
wrapper uses five singleton registrations, preserving the exact candidate IDs;
the four historical v1 cards retain their original identity metadata. The first
coordinator also spent time rebuilding memory after a completed job; it was
stopped and recovered against the same queue with batch publication. Completed
states were collected without restarting training or duplicating paid attempts.
Both preparation logs are archived. No training engine or test threshold changes
are part of this results PR.

Publication now advances the family selection card together with an explicit
view-policy advancement. It validates the old pins, stages the same candidate
choices against new verified rows, and archives the exact previous card with
its hash. Fully measured failures remain measurements; unsupported and
unmeasured recipes receive no qualification credit. Validation failures leave
the existing card and publication untouched. Regression tests cover this
revision mismatch, incomplete declarations, missing evidence, invalid old pins,
late validation failures, archive corruption and repeated cached refreshes.

Reproduce from the repository root in the project Python environment:

```sh
python reports/forge/word-split-inventory/run.py --stage plan
python reports/forge/word-split-inventory/run.py --stage run --gpus 0,1 \
  > runs/word-split-tier-results/controller.log 2>&1
tail -F runs/forge-word-split/technique-inventory-word-split-v1/progress.jsonl
python reports/forge/word-split-inventory/run.py --stage report
python reports/forge/gaussian-smoke-inventory/export_media.py \
  --queue-root runs/forge-word-split --attempts reports/forge/attempts \
  --output reports/forge/word-split-inventory/media \
  --campaign technique-inventory-word-split-v1
python reports/forge/word-split-inventory/publish.py
python reports/forge/regenerate_technique_inventory.py \
  --source-commit 737592c128ef84cf596c7f198b0ce8aad7c65700 --device cuda --advance-policy
python -m experiments.forge compile --summaries-only
python -m experiments.forge compile --check
```

The training replay needs the frozen executed scientific source. An existing
queue attaches to its exact frozen requests; incompatible declarations or
sources are refused. Publication alone uses the committed evidence snapshots
via `python reports/forge/regenerate_technique_inventory.py` and needs no training
or raw-artifact hydration. Independent regrading needs byte-exact original
receipts and their original paths; the display readout grants no qualification.
