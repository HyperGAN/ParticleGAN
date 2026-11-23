# Forge integration review map — 2026-10-02

Use the [original audit](../reports/forge-audit-2026-10-02.md) as the inspected
baseline at `b8feb1fe8e672c3e7a6b579329af0bd89a069ff0`. This map separates the
review obligations of the release integration from the focused preparation PRs.
It is an engineering completion map, not another leaderboard or new research
qualification. Preserve the [clean-MoG inventory](../reports/forge/technique-inventory.md)
and [served-policy inventory](../reports/forge/policy-family-inventory.md) as the
single authoritative board for each respective goal.

All six preparation PRs merged into `develop` on 2026-10-03 at 02:57:40 UTC
(2026-10-02 in America/Denver). The combined landed head is
[`81f0b91f2478d71f86aabeb83b43bca8c7f5c587`](https://github.com/HyperGAN/ParticleGAN/commit/81f0b91f2478d71f86aabeb83b43bca8c7f5c587).
Each focused head passed CI before the shared branch update. The merge includes
the safe summary-only memory refresh and preserves the original scientific
evidence, provenance pins and goal boards byte-for-byte.

## Review boundaries

| Boundary | Review entry points | Reviewer must establish | Relevant controls |
| --- | --- | --- | --- |
| Public trainer, recipes and state | `particlegan/training.py`, `particlegan/recipes.py`, priors, E22/Atlas policy/state modules; [develop integration](../reports/forge/DEVELOP_INTEGRATION.md) | Exercised recipe, prior, initialization, state, named RNG streams and sampling law match their own source cohort; saved-state continuation preserves the full policy | `test_forge_policy_integration.py`, `test_forge_boundaries.py`, `test_forge_sampling_boundary.py`, E22/Atlas tests |
| Executor and operational ownership | `experiments/forge/queue.py`, `worker.py`, `sources.py`, `runtime.py`; policy coordinator under `benchmarks/toy_audit/api_family_search.py` | Compatible submitters cause one physical launch; source is frozen; live leases fence recovery; complete reservations, interruption/retry costs and resource admission remain accounted | `test_forge_queue.py`, `test_forge_resources.py`, `test_toy_api_family_search.py` and policy ownership PR controls |
| Scientific scorers and protocol | `experiments/forge/views.py`, native/vector/image adapters, `benchmarks/toy_audit/api_contract.py`, task/view/calibration declarations | Full required denominators and unknowns remain; thresholds, terminal/hold windows, recipe/prior/runtime identities and clean/noisy serving cohorts are not exchanged | `test_toy_api_contract.py`, `test_forge_sampling.py`, calibration/lane/promotion controls |
| Evidence storage, reducers and publication | [artifact resolver PR #256](https://github.com/HyperGAN/ParticleGAN/pull/256); maintained `experiments/forge/policy_publication.py` and `policy_family_readout.py`; inventory publication wrapper | Exact originals are retrievable or explicitly missing; checksums, archive commit/blob/source identities and raw errors survive; reductions preserve measured costs and verdicts; publication grants no qualification | `test_forge_artifact_resolver.py` from #256; default-discovered `test_forge_policy_publication.py` including frozen-output parity |
| Agent recall and research decisions | `experiments/forge/knowledge.py`, planning/lifecycle/readout contracts; focused recall, calibration and decision PRs below | Every concluded trial is discoverable; publication freshness is visible; a proposed bounded measurement binds prior evidence, substantive delta, predicted numerical discriminator, falsifier and terminal action | Recall freshness/coverage tests, decision-contract controls, calibration feasibility controls |

An evidence-publication change can alter the implementation/source digest without
changing the recorded scientific result. Review that new identity explicitly;
do not transfer an old pass to a newer checkout or rerun unchanged science to
make a reporting merge look current. Frozen result receipts and provenance stay
the authority for their original cohort.

## Focused preparation PRs and acceptance

Each branch starts independently from the inspected `develop` base. Acceptance
records software controls and the combined integration rehearsal below; it does
not establish empirical calibration or access to unavailable archives.

| Audit priority | Focused change | Merged commit | Acceptance / remaining empirical work |
| --- | --- | --- | --- |
| P1: repair recall and freshness | [#259](https://github.com/HyperGAN/ParticleGAN/pull/259), `codex/forge-audit-recall` | [`c6c5a3dd`](https://github.com/HyperGAN/ParticleGAN/commit/c6c5a3dd2bd3d247222bed72f22fbe80496a3ce6) | Concluded Forge configurations and policy trials are normalized source-bound recall inputs; exact IDs/failed bounds/mechanisms/goals link authoritative evidence; stale/missing publication coverage is visible without training |
| P1: shared policy execution ownership | [#260](https://github.com/HyperGAN/ParticleGAN/pull/260), `codex/forge-audit-policy-ownership` | [`57f1b59a`](https://github.com/HyperGAN/ParticleGAN/commit/57f1b59ada2610d68437c0431eb7f8c451e4e62c) | Immutable execution source, coordinated deduplication/admission, inherited leases and independently supervised absolute deadlines; crashed callers cannot extend paid children, and completed originals resume certification without reexecution. GPU aliases are normalized; unresolved UUID/MIG masks fail before admission. No clean-MoG qualification from policy results |
| P1: retrieve original evidence | [#256](https://github.com/HyperGAN/ParticleGAN/pull/256), `codex/forge-audit-artifact-resolver` | [`694aaf3f`](https://github.com/HyperGAN/ParticleGAN/commit/694aaf3fd0ca4f75c84261897291dbe63be82b7f) | Verified fresh-process fixture hydration of full request/evidence/result/source and selected checkpoint; pinned Git originals; explicit invalid/missing outcomes and retention ownership gaps. Actual `/ml2` archives still require a real mounted/transferred copy |
| P1: prevent infeasible calibration spending | [#258](https://github.com/HyperGAN/ParticleGAN/pull/258), `codex/forge-audit-calibration-feasibility` | [`498a63ca`](https://github.com/HyperGAN/ParticleGAN/commit/498a63ca71de9a00e1858d56656064ef4a2b29f3) | Read-only logical feasibility preflight and registration guard preserve bound published matrices and original hashes. **A justified new screen, independent positive/negative references, and empirical accepted calibration remain pending** |
| P2: hypothesis-to-decision contract | [#262](https://github.com/HyperGAN/ParticleGAN/pull/262), `codex/forge-audit-decision-contract` | [`da5aed2b`](https://github.com/HyperGAN/ParticleGAN/commit/da5aed2b24438959322836125c4e0aa8d1b8ebbb) | New bare research ideas require reviewed v2 contracts: exact prior evidence, effective recipe/prior/component initialization, full task/job/runtime bindings, numerical prediction/falsifier, competing explanation and bounded stop/review rules. Missing, nonfinite, invalid or duplicate metrics remain incomplete. Exact legacy declarations and separately verified registered searches/lanes/promotions retain their original contracts; causal judgment remains research review |
| P2: maintain publication controls and make integration reviewable | [#257](https://github.com/HyperGAN/ParticleGAN/pull/257), `codex/forge-audit-publication` | [`81f0b91f`](https://github.com/HyperGAN/ParticleGAN/commit/81f0b91f2478d71f86aabeb83b43bca8c7f5c587) | Maintained reducers live in the experiment package; original negative controls enter default CI; frozen parity and deterministic bytes pass; original audit/reproduction identities and this review map are preserved |

The calibration preparation does not produce new data or an accepted screen.
Logical infeasibility can block wasteful matrix filling; engineering checks
cannot establish false-reject behavior, population reliability or adoption.
Existing published matrices retain their qualified source/initializer/recipe
bindings; projection or legacy-binding changes must not silently turn those
recorded outcomes into a newer cohort. Unknown independent-reference coverage
remains unknown. No new training, seed study, qualification regrade or promotion
is authorized by this map.

Execution ownership covers one cooperating Forge queue on one host. Campaign
budgets retain their own frozen definitions. A v2 decision's stable scientific
candidate-round identity additionally accumulates paid retries and live
reservations across campaign owners in that queue; narrative edits, metadata
revisions and job namespaces cannot reset its cap. Separate queue roots do not
provide global admission or spending guarantees. Use the same queue for these
shared ownership and bounded-round guarantees. A descriptive initialization
label alone is not proof that a changed mechanism was exercised.

## Landed integration and combined verification

The six independent review units were integrated and verified in a temporary
checkout before the combined update to `develop`. Historical rehearsal details
below describe that preparation, rather than pending merge instructions. Shared
CLI dispatch and catalog conflicts were resolved while preserving scientific
receipts; no qualification regrade or training run was used to complete the
merge.

The initial local rehearsal combined #256, #257, #259, #258 and #260. Its only
conflicts were catalog metadata and adjacent CLI early-return blocks. Preserve
both `artifacts` and `calibration-preflight` dispatch branches before any Queue
construction. For `configs/forge/catalog.json`, preserve the original top-level
fields and all `pinned_sources` entries, stage the resolved source changes so
`inventory()` records their actual Git blobs, regenerate inventory fields from
the combined tracked paths, then set coverage with `validate_inventory(root,
merged_catalog)`. Do not choose one branch's catalog wholesale or replace it
with `inventory()` alone. The five-PR rehearsal had valid 11,719/11,719 coverage;
the complete six-PR checkout must recount its own tracked paths. Coverage checks
path presence, not blob equality; compare the staged blob identities separately.

The final six-head rehearsal used #256 `95a30442`, #257 `1ae7749f`, #258
`6766617f`, #259 `778cb6f8`, #260 `d5a2a980`, and #262 `b7973457`. Its remaining
merge conflict was catalog metadata. The combined inventory covers
11,720/11,720 tracked paths, retains the three original `pinned_sources` entries,
and matches staged Git blob identities. The final documentation follow-up in
this PR changed no tested implementation. That rehearsal preceded the authorized
merge; the merged commit identities are recorded above.

For a later reporting or reducer-source change, refresh recall using
`python -m experiments.forge compile --summaries-only`, then verify
`python -m experiments.forge compile --check`.
The summary-only mode preserves published qualification and telemetry snapshots.
Plain `compile` is not the reporting-merge workflow: it can regrade historical
scientific inputs through the live reducer. A stale reducer fingerprint warrants
this safe refresh, not new qualification, reconstructed passes or paid reruns.

The final combined acceptance passed **1,333 software tests in 115.59 seconds**,
with eight existing multiprocessing-fork deprecation warnings:

```sh
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 MPLBACKEND=Agg python -m pytest -q \
  tests/test_forge*.py tests/test_toy_api_family_search.py
```

The safe summary refresh completed, and `compile --check` reports `CURRENT`
for 332 records, 75 published recall records and eight views. `validate` checks
49 tasks and eight views without training. Both inspected failed calibration
profiles remain `INFEASIBLE` from their bound published reports, with zero
binding issues; this preserves their original scientific decisions.

The combined diff preserves original attempts, experiment records, automation,
calibration reports/receipts, technique reports, configuration-search archives,
dated family-winner reproduction sources and scientific boards. Changes under
`reports/forge` are the safe compiled memory/recall/tier projections, declaration
pointers and publication migration/recall additions. Follow-up labels are
declaration-only pointers; their presence infers no outcome or new measurement.
The original audit and dated source bytes remain unchanged. These acceptance
results establish code preparation, not empirical accepted calibration or
availability of the missing originals.

Run the affected software suites and normal default CI discovery in the combined
checkout. Verify that read-only commands construct no worker/submission and that
publication controls remain included by `testpaths = ["tests"]`. Validation
must not train, hydrate nonexistent archives, regrade original matrices, fill
unknown cells, regenerate duplicate boards, or force-add ignored execution logs.
Raw stdout/JSONL/JUnit/checkpoint/state artifacts remain local; compact reports,
final metrics and provenance receipts remain reviewable in Git.

For [release PR #247](https://github.com/HyperGAN/ParticleGAN/pull/247), review the
public API/recipe behavior, executor invariants, scorer/protocol identities and
evidence source cohorts as separate obligations using the table above. Its
shipping selection rationale requires its applicable fully qualified winner,
accepted calibration and preregistered promotion evidence. Framework readiness
is a separate engineering decision. Focused preparation PRs can improve bounded
agent research without claiming a shipping default or changing #247's release
gate.

Completion of these six code priorities means the requested agent preparation is
implemented and reviewable; it does not mean an empirical accepted calibration
or the author's archived states now exist here. Accepted calibration still needs
a justified frozen successor and measured compatible references. Actual `/ml2`
archive retrieval remains pending until a real copy is configured. Release
qualification/adoption retain their separate original evidence requirements.

## Maintained publication entry points and frozen reproduction

Future maintenance belongs to:

```sh
python -m experiments.forge.policy_publication \
  --combined /path/to/existing/certified-combined.json \
  --output reports/forge --all-media
python -m experiments.forge.policy_family_readout \
  /path/to/existing/original-family-study.json \
  --output runs/forge/local-policy-readout
```

The publisher updates the existing policy inventory filenames. A local readout
is a display projection, not a second leaderboard. Inputs must resolve their
original receipt/artifact bindings; missing originals block publication. These
commands use saved verdict streams and certified bytes without training or gate
rescoring. No board was regenerated for this refactor.

The dated `reports/forge/family-winner-round1/publish_policy_results.py`,
`policy_family_readout.py`, and `test_publish_policy_results.py` remain
byte-identical frozen reproduction sources. New development uses the package
and `tests/test_forge_policy_publication.py`; retaining dated sources is an
intentional evidence exception, not a second maintained implementation. The
[migration provenance receipt](../reports/forge/publication-migration-2026-10-02.json)
records their exact baseline commit, Git blobs, byte hashes and maintained
destinations. Do not replace those originals with wrappers: archived publisher
contracts bind their bytes. The original audit is copied unchanged.

The package migration changes the publisher's implementation provenance,
package import and repository-root lookup. Maintained reducers also verify
explicitly retained original evidence origins after source-byte-identical reuse:
the actual original commit and runtime remain recorded, source bytes must match
the new study, and only placement may differ within the same hardware/runtime
cohort. A current-origin label cannot recertify a different original. Legacy
inputs retain their frozen behavior. Controls compare frozen and maintained
scientific projections and Markdown exactly, preserve receipt/artifact/source
identity, compare repeated JSON/media bytes deterministically, exercise the
readout CLI, check retained-origin/runtime reuse and refusals, and retain the
original source-bound negative cases. These are
synthetic publication controls, not new toy training evidence.

Tail local verification with `tail -F runs/forge/audit-publication/pytest.log`.
Default collection and focused test logs stay in the same ignored directory.
