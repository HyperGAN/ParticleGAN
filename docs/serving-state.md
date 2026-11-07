# Serving state and ownership

ParticleGAN selects a recipe's inference weights and sampling law. An application
decides when and how to publish those weights to its users. These are separate
responsibilities.

## Capture at an update boundary

Call `policy.served_snapshot()` or `policy.served_model()` before `begin_step()`
or after `finish_step()` returns successfully. The corresponding `GANTrainer`
methods have the same contract. An active update raises `RuntimeError` before
copying or swapping weights: its critic, generator, controls, averages and
`completed_steps` can describe different stages of that update.

Keep a previously captured `ServedModel` when inference is needed during an
update, including inside `generator_real`. Its frozen modules remain independent
of later training. After a failed update, recover from a completed checkpoint
before publishing another model; aborting bookkeeping does not undo applied
optimizer or particle moves.

The caller must serialize capture with updates to the same policy. The boundary
check is an ordering guard, not a lock for concurrent threads or processes.

## Three distinct kinds of state

| State | Owner and transition |
| --- | --- |
| Selected source | The policy derives `fast` or `averaged` from the recipe and its current eligibility evidence. Completing another update can change that choice. |
| Live parameter placement | Caller-owned loops retain fast weights. `GANTrainer` may expose averages between updates, retaining an ephemeral `_fast` backup. Training and checkpoint capture release that swap; checkpoints contain the fast iterate and averages. |
| Published inference copy | `served_snapshot()` returns independent tensors; `served_model()` builds frozen eval modules with a private evaluation stream. Later training or expiry changes future captures, not an existing copy. Sampling advances the copy's stream unless the caller supplies a separate generator. |

Selection always pairs generator, prior/table and any encoder/router from the
same source. The critic stays current. Ordinary module buffers follow the native
serving path; routed row buffers follow the selected table and represented mass.
Custom generation callbacks must use the supplied frozen modules; use
`generation_factory(models)` to bind auxiliary modules rather than closing over
live training parameters. See [the recovery and serving API](e22.md#recovery-and-serving).

## Eligibility evidence is local to the recipe

With averaging disabled, selection is always fast. With ordinary stationarity
control, an average is selected while the table's last decisive verdict is
stationary. The feature-cell backend instead requires its paired-average stamp
to be eligible, to match the current reaction serial, and to remain within
strictly less than one real-FIFO turnover. A full turnover expires it; a future
capture then selects fast weights. Checkpoint restore validates the saved source
against the saved evidence rather than treating the `served_source` label as an
independent authority.

That stamp is a bounded-stale anti-blur observation. It does not prove current
distribution quality at every update. Capturing, sampling, saving or restoring
does not renew it. Publishing an inference copy preserves what was selected at
capture; it does not promise that future training will remain eligible.

## Application responsibilities

The application owns synchronization, request routing, publication and rollback
of inference copies, storage, retention and any additional deployment or quality
gates. It also owns data cursors and stateful callback resources. ParticleGAN
provides no deployment registry, request lifecycle or cross-process serving lease.
Use the snapshot's `source` and `completed_steps` as provenance for that capture,
and retain the recipe and sampling options alongside exported artifacts. Clean
and output-noisy samples are distinct evaluation cohorts.
