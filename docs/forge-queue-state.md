# Forge queue state and ownership

Forge owns scheduling, process supervision, evidence collection and campaign
accounting. These are separate from the ParticleGAN API's training-update,
particle and serving state. Queue status does not alter a recipe, task, gate or
training checkpoint. A job's completion, scientific result, payment and readout
are separate facts; none is a substitute for another.

| Fact | Owner and transition | What it does not establish |
| --- | --- | --- |
| Submission | `Queue.submit`, `_refresh`, `cancel` and `retry` project one request's authorized scope into `queued`, `running`, `paused`, `blocked`, `completed`, `cancelled` or `concluded`. | A blocked or cancelled subscriber does not prove its shared job has stopped. `completed` is not scientific PASS or a concluded readout. |
| Shared job | `Queue.claim` reserves one compatible `pending` job as `running`; `collect` moves it to `terminal`; explicit execution repair can `retry` it as `pending`. | A terminal job can contain completed, failed, timed-out or cancelled execution, and separate task grades. |
| Worker attempt and lease | `Queue.launch` registers the process while holding the queue lock. `worker` supervises runner/grader children; its inherited execution descriptor fences recovery. | PID disappearance or an old heartbeat alone cannot authorize relaunch. Missing terminal receipt plus released execution lease permits orphan collection. |
| Result and grade | `Queue.collect` checks the attempt token and, when required, frozen evaluator certificate before publishing task rows. | Process success does not establish a task's PASS gate, comparability or qualification. |
| Reservation and charge | Claim records a payer and reserves its budget. Collection releases that reservation and charges that attempt once. Cancellation may retain or explicitly transfer the reservation to an authorized subscriber. | Subscriber withdrawal does not erase spent cost. A retry is a new attempt; it cannot move an earlier charge to its new payer. |
| Event delivery | `Queue.event` persists the outbox; `flush_events` appends/fsyncs target streams and deduplicates stable event IDs. | Delivery or receipt is not job completion or a scientific verdict. |
| Dedicated policy execution | `policy_execution` owns its separate `policy_attempts`, reservation and recovery contract; ordinary claims account for its active reservations. | It is not another ordinary queue submission and does not inherit submission projection rules. |
| Candidate/readout | `lifecycle` owns open/disposition checks and durable scientific readouts. | A scheduling projection does not conclude, abandon or reopen a candidate. |

## Submission projection contract

`Queue._refresh` excludes explicitly `cancelled` and `concluded` entries, then
uses the pure `_submission_projection` function in
[`experiments/forge/queue.py`](../experiments/forge/queue.py). Eligibility remains
owned by the frozen execution policy and tier/dependency checks. The projection
reads those observations in this order:

| First matching observation | Status | Lifecycle | Reason |
| --- | --- | --- | --- |
| Eligibility reports a blocker | `blocked` | `awaiting_readout` | The blocker |
| Authorized work is running | `running` | `running` | Cleared |
| No eligible or running work remains | `completed` | `awaiting_readout` | Cleared |
| Work remains and campaign is paused | `paused` | Retained | `campaign paused` |
| Otherwise | `queued` | Retained | Cleared |

Pause prevents future claims while already running workers retain their lease,
reservation and payer. A paused campaign can therefore show `running`, or
`completed` after its final attempt is collected. Queued and paused retain the
existing lifecycle because admission, a previous claim or explicit retry owns
that history. In particular, pausing between tasks can leave lifecycle
`running`. Neither status nor lifecycle is a process-liveness probe.

Closed submissions remain closed under refresh. An explicit re-enqueue can
resubscribe a cancelled request; execution retry has its existing authorization,
attempt limit and provenance checks. This projection introduces no new states,
retry paths, source compatibility rules, lease policy or accounting transitions.

The enumerated precedence table and pause/collection/accounting integration are
covered by [`tests/test_forge_queue_state.py`](../tests/test_forge_queue_state.py).
The existing queue, worker, decision-contract and
[external-lease tests](../tests/test_forge_shared_leases.py) cover shared
subscribers, transfer limits, descendant fencing and descriptor lifetime. See
[external fleet leases](forge-shared-leases.md) for the separate descriptor
ownership contract and retained STOP on the archived external runner.
