# Reusable candidates, task gates and research studies

Forge separates three declarations. A candidate defines training mechanisms,
hyperparameters, capability requirements and claims. A task owns data,
architecture, initialization, prior, sampling, training allowance, metrics,
thresholds and sustained PASS/FAIL. A study chooses the candidate and control,
states the hypothesis and predictions/falsifiers, cites original evidence,
and freezes one finite campaign with stopping rules.

| Declaration | Authoritative fields | Generated output |
| --- | --- | --- |
| Candidate (`ideas/` or `configurations/`, schema 3) | Global recipe, mechanisms, capability requirements, compatibility claims | Resolved public recipe and source-bound candidate revision |
| Task (`tasks/`, existing schema) | Host/data/model, initialization, prior, resources/updates, sampling, evaluator and sustained thresholds | Execution/evaluation fingerprints and ownership receipt |
| Study (`studies/`, schema 1) | Candidate/control selection and task mapping, hypothesis, view/tier/backend, prior-evidence selectors, prediction/falsifier, finite campaign and terminal rules | Task membership, exact exercised delta, evidence hashes, source/protocol/runtime/job bindings and admission receipt |

A study does not redefine task criteria or supply recipe overrides. Its campaign
is the only authored definition of campaign and candidate caps. The task's
`resources.timeout_seconds` remains the full per-job reservation allowance;
`execution.steps` remains the training limit. View policy selects the required
denominator. Scientific gates do not consume predictions.

New candidates reject study fields, explicit prior conditions and task-owned
resource overrides. An optional candidate initializer is a compatibility
requirement, not an override of task initialization. Existing field ownership,
public API preflight, fixed named streams and source/runtime cohorts still apply.
Historical host optimizer constants remain labelled inactive provenance; active
training values come from the resolved candidate/task binding.

## Complete small example

The example uses one pure BCAP recipe, the existing scalar Gaussian task and
one ordinary Tier 1 study. The current view still requires all six Tier 1 tasks;
showing one task here does not shrink that denominator. The complete Tier 1
scope also includes the existing 300-second clock/state diagnostic, so this
example reserves 2,520 seconds including it. This is a declaration
example, not a measured result, recommended tuning or permission to train.
The numerical prediction is illustrative and the example stays draft.

The recipe below preserves every override, capability and claim from the pure
BCAP draft at [b509c065](https://github.com/HyperGAN/ParticleGAN/blob/b509c0654b91d9fce9b49374dde02ec75c1bd625/configs/forge/ideas/bcap-pure-adam-v1.json).
Plain Adam, fixed coefficient/cap/every-update BCAP, and the disabled clipping,
A2, anchoring, direct-particle response, extra regularization and additive
training noise are unchanged. The task owns the learned MoG conditions.
The imported candidate has a new schema/execution identity; no old draft or
scientific receipt is rewritten. Its original v2 declaration is retained byte for
byte in `tests/fixtures/forge/pure-bcap-b509c065.json`: original blob
`a586dc3b0cecb81e2d3799b753a2fd3e91b9e82f`, SHA-256
`f9e291389fb2207fb701696b4f3ae17c185f4eab60864f69a92977fa5979d7f6`.
The committed `pure-bcap-baseline-v1` study remains draft with its original
unresolved prediction/evidence intent. Existing behavioral host-owned settings
can refuse this exact pure recipe; those cells remain BLOCKED and no override
is removed to make them run.

Candidate: [`configs/forge/ideas/bcap-pure-adam-example-v3.json`](../configs/forge/ideas/bcap-pure-adam-example-v3.json).

```json
{
  "api_changes": [],
  "api_version": "forge-api-v1",
  "changed_factors": [
    "Native Adam without critic clipping, A2 prior damping, direct-particle response or EMA anchor",
    "Fixed BCAP coefficient1/cap1/every-update with constant learning rates and moments",
    "Zero additive training input/output noise; no prior regularizer or EMA selection"
  ],
  "claim_contract": {
    "sampling_law": "task_declared",
    "schedule": "scheduled",
    "scoring_weights": "live"
  },
  "execution_path": "public_trainer",
  "guide": "EXPERIMENTATION.md",
  "id": "bcap-pure-adam-example-v3",
  "mechanism_class": "structural",
  "mechanism_rationale": "Pure BCAP training baseline: public relativistic-pairing logistic loss plus the fixed one-sided L2 critic-input gradient cap on real and generated samples, updated by native Adam. This is a complete baseline definition, not a single-factor causal ablation.",
  "parent": "k3p-bcap-matched-v1",
  "prior_art": [
    "k3p-bcap-matched-v1"
  ],
  "recipe_overrides": {
    "beta2_end": null,
    "betas": [
      0.0,
      0.999
    ],
    "d_guard_ratio": 0.0,
    "d_lr_mult": 1.0,
    "direct_particle_gain": false,
    "ema_decay": 0.0,
    "input_noise_std": 0.0,
    "latent_damping_max_rate": 0.0,
    "lr": 0.00425,
    "lr_floor": 1.0,
    "name": "bcap-pure-adam-v1",
    "network_lr_floor": 1.0,
    "network_lr_horizon_cap": null,
    "optimizer_family": "adam",
    "output_noise_std": 0.0,
    "output_noise_warmup": 0.0,
    "prior_lr_mult": 2.0,
    "prior_reg": 0.0,
    "reg_anchor_weight": 0.0,
    "reg_arm": "b_cap",
    "reg_coeff": 1.0,
    "reg_coeff_end": null,
    "reg_every": 1,
    "reg_kappa": 1.0
  },
  "requires_capabilities": [
    "learned_locations",
    "mog_prior",
    "named_rng"
  ],
  "schema_version": 3
}
```

Task: [`configs/forge/tasks/gaussian1d_acquisition.json`](../configs/forge/tasks/gaussian1d_acquisition.json).
This example uses the current recipe-owned prior contract. Archived declarations
retain their original task-owned prior policy. Its sustained gate
requires five terminal passing checks out of 24 observations, including location,
width and CDF shape; a good predicted final location alone cannot PASS it.

```json
{
  "schema_version": 1,
  "id": "gaussian1d_acquisition",
  "adapter": "transfer_vector",
  "execution": {
    "initializer": "deterministic_orthogonal",
    "host": "gaussian1d_acquisition",
    "steps": 1000,
    "prior_contract": "recipe_owned_v1",
    "prior": {
      "kind": "mog",
      "sigma": 0.025,
      "standardize": false
    },
    "protocol": "screening",
    "produces_state": false,
    "host_source": "benchmarks/toy_audit/gaussian1d_quality.py",
    "host_definition": {
      "hidden": 32,
      "layers": 2,
      "fourier": 2,
      "z_dim": 2,
      "particles": 256,
      "batch": 128,
      "steps": 1000,
      "lr": 0.00425,
      "d_lr_mult": 1.0,
      "prior_lr_mult": 2.0,
      "prior_reg": 0.0,
      "betas": [
        0.0,
        0.999
      ],
      "ema_decay": 0.995,
      "reg_arm": "k3p",
      "reg_coeff": 1.0,
      "reg_kappa": 1.0,
      "d_every": 1,
      "g_every": 1,
      "family": "gaussian1d_acquisition",
      "split": "development",
      "kind": "gaussian_mixture",
      "means": [
        [
          2.0
        ]
      ],
      "covariances": [
        [
          [
            0.25
          ]
        ]
      ],
      "masses": [
        1.0
      ],
      "identifiable": true
    }
  },
  "evaluation": {
    "sampling_contract_version": 1,
    "eval_output_noise": "clean",
    "kind": "transfer_sustained",
    "evaluator": "benchmarks.transfer_suite.protocol:test_verdict",
    "gate_policy": {
      "id": "gaussian1d-acquisition-v1",
      "finite_atom_exemptions": false,
      "rationale": "Provisional scalar acquisition gate fixed before training. Exact analytic CDF, location and width reject point collapse, shift, wrong spread and discrete same-moment impostors; independent oracle controls do not calibrate the tier."
    },
    "sampling_law": "public_prior_without_output_noise",
    "thresholds": [
      [
        "sample_count",
        ">=",
        4096
      ],
      [
        "finite_fraction",
        "==",
        1.0
      ],
      [
        "mean_error_sigma",
        "<=",
        0.2
      ],
      [
        "std_ratio",
        ">=",
        0.8
      ],
      [
        "std_ratio",
        "<=",
        1.2
      ],
      [
        "cdf_ks",
        "<=",
        0.05
      ]
    ],
    "observations": 24,
    "minimum_stable_checks": 5,
    "scoring_weights": "live",
    "sources": {
      "benchmarks/transfer_suite/protocol.py": "99469b022b790a18a74021a6fe49424d95f535afaa220643688a1ddd7a70ab89",
      "benchmarks/locked_shared/observation.py": "bd6f9845b44f1ec2a58d445727990ba5068c7aca3b6f981cf38d738a37c4513b",
      "benchmarks/transfer_suite/vector_tasks.py": "3ee4eb27759f61a430029c80b7772cac3dca2fb2ac84919ace0db94efca1e0b0",
      "benchmarks/toy_audit/gaussian1d_quality.py": "7ec72e07c1aea87e77c85401b7e822f23d5a248ba45d718180045a1ad64ccd8b"
    },
    "sample_evaluator": "benchmarks.toy_audit.gaussian1d_quality:score_samples"
  },
  "resources": {
    "gpus": 1,
    "gpu_memory_mb": 2048,
    "cpu_threads": 1,
    "timeout_seconds": 120
  },
  "requires_capabilities": [
    "checkpoint",
    "named_rng",
    "live_sampling",
    "learned_locations",
    "mog_prior"
  ],
  "dependencies": [],
  "description": "Can the public ParticleGAN trainer acquire the scalar law N(2, 0.5^2) from random initialization within 1,000 updates, with correct location, width and CDF shape at five terminal checks?",
  "research_artifacts": {
    "api_publication": "reports/toy_audit/api_contract/gaussian1d/results.json",
    "readout": "reports/toy_audit/api_contract/gaussian1d/README.md"
  },
  "retained_question_ids": [
    "develop-gaussian1d_acquisition"
  ]
}
```

Study: save the following as `configs/forge/studies/pure-bcap-example.json`.
Evidence uses the original record identity as motivation; it grants no new
qualification. No candidate, source, job, runtime or evidence SHA is authored.

```json
{
  "schema_version": 1,
  "id": "pure-bcap-example",
  "status": "draft",
  "candidate": "bcap-pure-adam-example-v3",
  "control": {
    "candidate_id": "k3p-bcap-matched-v1",
    "task_map": {}
  },
  "hypothesis": "A plain-Adam relativistic-logistic GAN with only the fixed real/fake BCAP penalty may acquire the declared Tier 1 distributions without optimizer interventions.",
  "scope": {
    "view": "discriminator_stability",
    "through_tier": 1,
    "execution_backend": "cpu"
  },
  "campaign": {
    "id": "pure-bcap-example",
    "candidate_budget_seconds": 2520,
    "budget_seconds": 2520
  },
  "max_rounds": 1,
  "prior_evidence": [
    {
      "path": "reports/forge/records/readout-21b533460c29dd5049aa64df.json",
      "selector": [],
      "identity": {
        "record_id": "readout-21b533460c29dd5049aa64df",
        "candidate_id": "k3p-bcap-matched-v1"
      },
      "use": "motivation_only"
    }
  ],
  "prediction": {
    "task_id": "gaussian1d_acquisition",
    "metric": "mean_error_sigma",
    "op": "<=",
    "threshold": 0.2,
    "phase": "final"
  },
  "falsifier": {
    "task_id": "gaussian1d_acquisition",
    "metric": "mean_error_sigma",
    "op": ">",
    "threshold": 0.2,
    "phase": "final"
  },
  "competing_explanation": "Removing several interventions together defines a baseline but cannot attribute a pass or failure to one removed mechanism.",
  "terminal_rules": {
    "falsified": "stop_revision",
    "prediction_observed": "review_saved_diagnostics",
    "inconclusive": "stop_and_readout",
    "incomplete": "request_missing_evidence"
  }
}
```

## Prepare, plan and enqueue

Read `AGENTS.md`, `EXPERIMENTATION.md` and compiled memory first. To scaffold a
new recipe and a companion draft study:

```sh
python -m experiments.forge new --id YOUR_CANDIDATE --parent EXISTING_CANDIDATE \
  --goal discriminator_stability --study-id YOUR_STUDY \
  --hypothesis "State the bounded research question"
```

Edit training mechanisms and `changed_factors` in the candidate. Edit selection,
hypothesis, original evidence path/selector/identity, numerical signatures,
competing explanation, scope and finite campaign in the study. Empty `task_map`
means the same declared tasks for the control; an explicit mapping must cover
every authorized task. Neither controls nor prior evidence silently launch runs.

```sh
python -m experiments.forge plan bcap-pure-adam-example-v3 --study pure-bcap-example \
  --show-boundaries
```

Planning writes nothing and trains nothing. Inspect `study_binding.actual_bindings`
and `expected.substantive_delta`. They use the same public task resolver as
execution. Review the generated cohort/task/job bindings and full reservation
ceiling. Finish the draft and set its study `status` to `ready`. Seed-only,
prose-only, inactive or descriptor-only changes, unsafe/mismatched evidence,
unsupported scope and insufficient caps cannot obtain admission by setting ready.

```sh
# Explicit submission freezes source and study bindings; starts no workers.
python -m experiments.forge enqueue bcap-pure-adam-example-v3 --study pure-bcap-example
python -m experiments.forge logs --follow --candidate bcap-pure-adam-example-v3
```

The study owns its campaign; omit `--campaign`. Optional view/tier/device/model
flags must agree with its declared scope. `run ... --study ... --gpus ...` remains
an explicit training action and must match the study's compute cohort.
Enqueue independently re-resolves trusted declarations and checks source bytes,
runtime, recipe, task gates, grouped jobs and the generated receipt before queue
mutation. The queue freezes the study ID and binding. Changing an admitted study
or its source/task/runtime requires another study ID; the same scientific round
cannot reset its cap through a new ID, narrative or campaign. Full reservations,
paid retries and shared jobs count under the existing queue-wide round accounting.

The compiled internal decision projection is generated compatibility data. It is
never stored in the candidate or edited as a second authoritative declaration.
Drafts and missing study bindings block new recipe admission. Bare configuration
IDs, schema downgrades and removal of a legacy embedded contract grant no bypass.

## Use the same candidate in another study

```sh
python -m experiments.forge study new --id pure-bcap-second-question \
  --candidate bcap-pure-adam-example-v3 --control k3p-bcap-matched-v1 \
  --view discriminator_stability --device cpu \
  --hypothesis "State another bounded question for the same recipe"
python -m experiments.forge plan bcap-pure-adam-example-v3 --study pure-bcap-second-question
```

Only the new study file is created. It can select another allowed view/tier,
control or prediction while using the byte-identical candidate. It starts draft
and needs the same evidence, finite scope and review. Study IDs change request
identity, not scientific job compatibility: exact recipe/task/source/runtime
jobs can share work and receipts. Different priors, sampling laws or sources
remain separate scientific cohorts. No unchanged experiment is automatically
rerun and no new study constitutes independent replication.

`tests/test_forge_studies.py` demonstrates this end to end with two studies,
different scopes, budgets and predictions, one unchanged candidate, shared queue
jobs and separate frozen readouts. It uses synthetic software receipts and runs
no training. Ordinary qualification still uses the full view and sustained gates.

## Read out the frozen study

```sh
python -m experiments.forge readout bcap-pure-adam-example-v3 --study pure-bcap-example \
  --conclusion "State the measured result and limits" \
  --comparison "Compare the original compatible control evidence" \
  --next-action "Stop or name the next separately bounded question"
```

Readout uses the saved queue request and compatible original receipts, not the
current candidate/study JSON. Each study gets a separate record identity and
prediction outcome. A prediction can be observed while scientific gates FAIL.
Missing, nonfinite, invalid or conflicting evidence remains incomplete. A
falsifier takes precedence; outcomes permit only stop/readout/review actions,
never automatic continuation or qualification. Shared receipts retain producer
request/study IDs and original hashes/costs; overlapping readouts are not summed
as additional training or independent experiments. Concluding one study does
not close another use of the recipe. Study readout refreshes compact memory
without regrading historical qualification snapshots or creating another goal
leaderboard.

## Compatibility and searches

Original v1 cards use the pinned legacy admission manifest. Original v2 cards
retain their embedded contract reader and frozen hashes. Do not delete a contract
or replace it with a new study on the old candidate: declare an explicit v3
successor. Saved requests, receipts and historical readouts require no migration
or current source reconstruction. No historical card/result is rewritten by this
refactor. The [historical decision guide](forge-decision-contract.md) documents
that format.

New [configuration-search specs](forge-configuration-search.md) use schema 2.
Their finite grid, hypothesis, tuning scope, deterministic selection and campaign
remain study responsibilities; protocol hashes and request bindings are generated.
Trials are reusable v3 configurations and do not inherit an embedded decision
contract or first-study hypothesis/report binding. Admission still verifies the
exact finite registered grid, mechanism signature, active axes, complete budgets,
source/runtime and candidate/task/job identities. Existing schema-1 studies and
configuration hashes retain their original semantics. A new search request also
freezes its study declaration; `readout CONFIGURATION --study SEARCH_ID` uses
that saved hypothesis and scope, including compatible reused receipts. Search selection and
prediction readouts cannot adopt defaults or bypass calibration.
