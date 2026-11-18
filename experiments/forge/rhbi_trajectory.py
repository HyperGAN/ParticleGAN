"""Passive, study-scoped archive of the executed marginal RHBI cross-plan."""
import hashlib
from pathlib import Path
from unittest.mock import patch

import torch

from .artifacts import manifest_artifacts

VIEW = "rhbi-retention-trajectory-v1"


def capture_steps(budget):
    # Early single-update resolution, then regular intervals and all official observations.
    return {0, *range(1, min(budget, 32) + 1), *range(10, budget + 1, 10),
            *((i * budget + 23) // 24 for i in range(1, 25))}


def attach(request, trainer, context, task, output):
    view = (request or {}).get("view")
    if not isinstance(view, dict) or view.get("id") != VIEW:
        return None
    if (task["id"] not in {"vector_unequal_mass", "interp_sparse_arc"} or
            request["candidate"]["id"] not in {"rhbi-v12", "rhbi-v12-feature-transport"} or
            view.get("evidence_scope") != "research_diagnostic"):
        raise ValueError("The RHBI trajectory archive requires its declared research arms and tasks")
    return TrajectoryArchive(trainer, context, task, Path(output) / "rhbi-trajectory")


class TrajectoryArchive:
    def __init__(self, trainer, context, task, directory):
        self.trainer = trainer
        self.context = context
        self.directory = directory
        self.directory.mkdir()
        self.budget = task["execution"]["steps"]
        self.milestones = capture_steps(self.budget)
        self.batches = hashlib.sha256()
        self.rows = []
        self.steps = 0
        self.save_state(0)

    def save_state(self, step):
        torch.save(self.context.state_dict(), self.directory / f"state-{step:04d}.pt")

    def step(self, real):
        """Observe just the generator's updating marginal forward. The solver returns its original tensor."""
        trainer = self.trainer
        core = trainer.loss.rhbi
        forward, prior = core.forward, trainer._sample_training_prior
        pair, solve = core._born_infeld_pair, core._sinkhorn_slack
        draws, observations = [], []

        def sample(n):
            latent, indices = prior(n)
            draws.append(indices.detach())
            return latent, indices

        def marginal(x, y, *args, **kwargs):
            if not kwargs.get("allow_ell_update", False):
                return forward(x, y, *args, **kwargs)
            ell = []
            plans = []

            def cost(a, b, horizon):
                ell.append(horizon.detach())
                return pair(a, b, horizon)

            def sinkhorn(hat, row_logits, col_logits, **options):
                plan = solve(hat, row_logits, col_logits, **options)
                if not plans:
                    # Exact counterfactual on this call: keep local horizons, centered
                    # logits, slack and masks; replace only the cross-coordinate cost.
                    with torch.no_grad():
                        _, raw_hat = pair(x.detach(), y.detach(), ell[0])
                        raw_plan = solve(raw_hat, row_logits, col_logits, **options)
                        if not trainer.loss.feature_transport and not torch.equal(raw_plan, plan):
                            raise ValueError("Raw-cost replay differs from the executed parent plan")
                        active_rows = plan[:-1].detach()
                        raw_rows = raw_plan[:-1]
                        tv = (active_rows.double() - raw_rows.double()).abs().sum() / (active_rows.double().sum() + raw_rows.double().sum())
                        observations.append({"plan": active_rows.cpu(), "direct_tv": tv.cpu(),
                                             "generator_rows": draws[-1].cpu(), "real": y.detach().cpu()})
                plans.append(True)
                return plan

            with patch.object(core, "_born_infeld_pair", cost), patch.object(core, "_sinkhorn_slack", sinkhorn):
                result = forward(x, y, *args, **kwargs)
            if len(plans) != 2 or len(ell) != 2:
                raise ValueError("Expected one cross and one self solve in marginal generator transport")
            return result

        with patch.object(trainer, "_sample_training_prior", sample), patch.object(core, "forward", marginal):
            result = trainer.step(real, collect_stats=True)
        if len(observations) != 1 or len(draws) != 2 or trainer.completed_steps != self.steps + 1:
            raise ValueError("Missing or ambiguous executed RHBI generator plan or latent draw")
        self.steps += 1
        self.batches.update(real.detach().cpu().contiguous().numpy().tobytes())
        self.rows.append(observations[0])
        if self.steps in self.milestones:
            self.save_state(self.steps)
        if self.steps % 50 == 0 or self.steps == self.budget:
            chunk = {key: torch.stack([row[key] for row in self.rows]) for key in self.rows[0]}
            torch.save(chunk, self.directory / f"plans-{self.steps:04d}.pt")
            self.rows.clear()
        return result

    def receipt(self):
        if self.steps != self.budget or self.rows:
            raise ValueError("Incomplete RHBI trajectory")
        return {"schema_version": 1, "directory": self.directory.name, "state_steps": sorted(self.milestones),
                "plan_updates": self.steps, "real_batch_sequence_sha256": self.batches.hexdigest(),
                "artifacts": manifest_artifacts(self.directory), "prerequisite_eligible": False}
