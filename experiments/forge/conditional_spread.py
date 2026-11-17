"""Wide continuous conditional GAN, using the released public EMPF component loop."""
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from benchmarks.toy_audit.conditional_spread import BAGS, BAG_SIZE, SCALES, WIDTH, score_samples, target_signals
from particlegan import UpdatePolicy
from particlegan.empf import bind_empf_record
from particlegan.optim.constraint_geometry import constraint_geometry_backward

from .api import CapabilityError, task_formulation_context
from .artifacts import save_provenance_checkpoint
from .contracts import atomic_json, file_hash
from .mechanisms import MechanismAudit, mechanism_blockers
from .priors import prior_policy_receipt
from .sampling import PUBLIC_PRIOR_CLEAN, executed_receipt


class ConditionedMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim + 1, hidden), nn.LeakyReLU(.2),
                                 nn.Linear(hidden, hidden), nn.LeakyReLU(.2), nn.Linear(hidden, out_dim))

    def forward(self, rows, group):
        return self.net(torch.cat((rows, group), dim=1))


def context_for(candidate, task, protocol, device, root=None):
    host = task["execution"]["host_definition"]
    expected = dict(width=WIDTH, group_sds=SCALES.tolist(), bag_size=BAG_SIZE, bags_per_group=BAGS)
    if any(host.get(key) != value for key, value in expected.items()):
        raise ValueError("conditional spread data law differs from executable target")
    evaluation = task["evaluation"]
    if (evaluation["eval_samples"] != 2 * BAGS * BAG_SIZE or evaluation["observations"] != 24
            or evaluation["minimum_stable_checks"] != 5 or evaluation["scoring_weights"] != "live"):
        raise ValueError("conditional spread evaluation differs from its live 24-observation protocol")
    if root is not None:
        for path, expected_hash in evaluation["sources"].items():
            if file_hash(Path(root) / path) != expected_hash:
                raise ValueError(f"conditional spread source binding differs: {path}")
    ctx = task_formulation_context(candidate, task, protocol, device=device, root=root)
    r = ctx.recipe
    if (ctx.streams.seed != 0 or ctx.initializer != "deterministic_orthogonal"
            or ctx.prior_mechanism is not None or ctx.prior_config["kind"] != "mog"
            or r.effective_critic_formulation != "empf" or r.prior_regularizer != "empf"
            or r.model != "gan" or r.encoder_mode != "none" or r.conditioning != "scalar"
            or r.ema_decay or r.output_noise_std or r.input_noise_std or r.latent_damping_max_rate
            or r.reg_every != 1 or r.reg_coeff <= 0 or r.prior_update != "learned"
            or r.constraint_geometry_mode not in ("none", "direction_blend")
            or r.generator_gradient_response != "none" or r.training_precision not in ("float32", "bf16_channels_last")):
        raise CapabilityError(["wide conditional host currently binds clean EMPF, learned MoG, seed 0 and reg_every=1"])
    return ctx


class Game:
    def __init__(self, ctx, task):
        self.ctx = ctx
        self.task = task
        self.recipe = r = ctx.recipe
        hidden = task["execution"]["host_definition"]["hidden"]

        def model(inputs, outputs, name):
            net = ctx.construct(lambda: ConditionedMLP(inputs, outputs, hidden), component=name)
            return ctx.initialize(net.to(ctx.device), component=name)

        self.G = model(r.z_dim, WIDTH, "generator")
        self.D = model(WIDTH, 1, "discriminator")
        self.prior = ctx.build_prior()
        self.opt_g, self.opt_d = r.make_optimizers(self.G, self.D, self.prior)
        self.penalty = r.make_critic_penalty(self.opt_d, input_domain="continuous", collect_stats=True)
        self.loss = r.make_loss()
        self.reg = r.make_prior_regularizer(weight=1. if r.prior_reg > 0 else 0.,
                                            prior=self.prior, reference=self.prior)
        bind_empf_record(self.opt_d.record, self.loss, self.reg)
        self.policy = UpdatePolicy(r, self.G, self.D, prior=self.prior, row_semantics="conditional",
                                   generator_optimizer=self.opt_g, critic_optimizer=self.opt_d,
                                   penalty=self.penalty, seed=ctx.streams.seed,
                                   streams={name: self.rng(family, name) for name, family in
                                            (("latent_generator", "prior"), ("penalty_generator", "noise"),
                                             ("noise_generator", "noise"), ("eval_generator", "eval"))})
        self.scales = torch.tensor(SCALES, dtype=torch.float32, device=ctx.device)

    def rng(self, family, component):
        return self.ctx.streams.generator(family, component=component)

    def batch(self, role):
        b = self.recipe.batch_size
        gen = self.rng("data", role)
        group = torch.arange(b, device=self.ctx.device) % 2
        group = group[torch.randperm(b, generator=gen, device=self.ctx.device)]
        rows = self.scales.new_zeros(b, WIDTH)
        rows[:, :2] = torch.randn((b, 2), generator=gen, device=rows.device) * self.scales[group]
        return rows, group[:, None].float()

    def generate(self, group, latent, kernel):
        z, idx = self.prior.sample(len(group), generator=latent, noise_generator=kernel)
        return self.G(z, group), z, idx

    def step(self):
        real, group = self.batch("critic")
        self.policy.begin_step(real, game_record=self.opt_d.record, execution_limit=self.task["execution"]["steps"])
        try:
            with torch.no_grad():
                fake, z, _ = self.generate(group, self.rng("prior", "critic"), self.rng("noise", "critic_kernel"))
                self.policy.observe_support(z)
            self.policy.observe_critic_pair(real, fake)
            d_loss = self.loss.d_loss(self.D(real, group), self.D(fake, group))
            penalty = self.penalty(self.D, real, fake, group)
            self.opt_d.zero_grad()
            (d_loss + penalty).backward()
            self.opt_d.step()
            self.policy.after_critic_step()

            real, group = self.batch("generator")
            self.D.requires_grad_(False)
            fake, _, idx = self.generate(group, self.rng("prior", "generator"), self.rng("noise", "generator_kernel"))
            self.policy.observe_sampled_rows(idx)
            g_loss = self.loss.g_loss(self.D(fake, group), self.D(real, group).detach())
            prior_loss = self.recipe.prior_reg * self.reg(self.prior.z)
            if self.recipe.prior_l2:
                prior_loss = prior_loss + self.recipe.prior_l2 * self.prior.z.square().mean()
            self.opt_g.zero_grad()
            constraint_geometry_backward(g_loss + prior_loss, self.opt_g, (g_loss,))
            self.policy.after_generator_backward(loss_gan=g_loss.detach(), loss_critic=d_loss.detach())
            self.opt_g.step()
            self.policy.after_generator_step()
            self.policy.finish_step()
        except Exception:
            self.policy.abort_step()
            raise
        finally:
            self.D.requires_grad_(True)
        if not torch.isfinite(torch.stack((d_loss, penalty, g_loss, prior_loss))).all():
            raise FloatingPointError("nonfinite conditional GAN update")

    @torch.no_grad()
    def samples(self):
        # Fixed evaluation panels couple observation noise across checkpoints.
        generators = [torch.Generator(device=self.ctx.device).manual_seed(
            self.ctx.streams.seed_for("eval", component=name)) for name in ("spread_latent", "spread_kernel")]
        groups = []
        for label in (0., 1.):
            condition = self.scales.new_full((2048, 1), label)
            chunks = [self.generate(condition, *generators)[0].cpu()
                      for _ in range(BAGS * BAG_SIZE // len(condition))]
            groups.append(torch.cat(chunks).numpy())
        return np.stack(groups)

    def state_dict(self):
        return dict(task=self.task["id"], recipe=self.recipe.to_dict(), policy=self.policy.state_dict(),
                    models=[m.state_dict() for m in (self.G, self.D, self.prior)],
                    optimizers=[o.state_dict() for o in (self.opt_g, self.opt_d)],
                    streams=self.ctx.streams.state_dict(), **self.recipe.prior_regularizer_checkpoint(self.reg))

    def load_state_dict(self, state):
        if state["recipe"] != self.recipe.to_dict() or state["task"] != self.task["id"]:
            raise ValueError("conditional spread checkpoint does not match task and recipe")
        for model, saved in zip((self.G, self.D, self.prior), state["models"], strict=True):
            model.load_state_dict(saved)
        for opt, saved in zip((self.opt_g, self.opt_d), state["optimizers"], strict=True):
            opt.load_state_dict(saved)
        self.ctx.streams.load_state_dict(state["streams"])
        self.recipe.restore_prior_regularizer(self.reg, state.get("prior_regularizer"))
        self.policy.load_state_dict(state["policy"])


def run(request, task, output, device):
    from benchmarks.toy_audit.api_run import render_gif
    from .adapters import _checkpoints, _event, _finite_tree

    start = time.monotonic()
    ctx = context_for(request["candidate"], task, request["protocol"], device)
    game = Game(ctx, task)
    audit = MechanismAudit(ctx.recipe, game.opt_d, [game.opt_g])
    steps = task["execution"]["steps"]
    checkpoints = set(_checkpoints(task))
    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)
    rows, frames = [], []
    deviations = 0
    reference = target_signals(102).reshape(2, -1, 2)
    for step in range(1, steps + 1):
        game.step()
        audit.observe_penalty(game.penalty.last_stats)
        if step in checkpoints:
            before = ctx.streams.audit()
            samples = game.samples()
            metrics = score_samples(samples)
            deviations += ctx.streams.compare(before, ctx.streams.audit())["unintended_rng_deviations"]
            rows.append(dict(step=step, **metrics))
            failed = [key for key, op, bound in task["evaluation"]["thresholds"] if metrics[key] > bound]
            views = [dict(kind="scatter", title=f"Group {group}", target=reference[group, :256],
                          samples=samples[group, :256, :2]) for group in (0, 1)]
            frames.append(dict(step=step, metrics=metrics, passed=not failed, failed_bounds=failed, views=views))
            _event("conditional_spread_observation", step=step, coverage=metrics["combined_coverage"])
    path = output / "goal.gif"
    render_gif(dict(id=task["id"], goal=task["description"], default_steps=steps), frames, path,
               full_budget=True, requested_steps=steps,
               final_verdict="PASS" if all(f["passed"] for f in frames[-5:]) else "FAIL")
    state = game.state_dict()
    counts = {role: min(int(opt.state[p].get("step", 0)) for p in module.parameters())
              for role, module, opt in (("generator", game.G, game.opt_g), ("discriminator", game.D, game.opt_d),
                                        ("prior", game.prior, game.opt_g))}
    mechanism = audit.receipt()
    evidence = dict(observations=rows, live=rows[-1], prior_policy=prior_policy_receipt(ctx.recipe, game.prior),
                    actual_training_gif=dict(path=path.name, sha256=file_hash(path)),
                    penalty_calls=game.opt_d.record.calls,
                    guards=dict(all_finite=_finite_tree(state), optimizer_updates=counts,
                                unintended_rng_deviations=deviations, mechanism_audit=mechanism,
                                hooks_exercised=not mechanism_blockers(mechanism) and
                                game.opt_d.record.calls == game.opt_d.record.observed_steps == steps),
                    provenance_checkpoint=save_provenance_checkpoint(output, state, completed_steps=steps),
                    **executed_receipt(PUBLIC_PRIOR_CLEAN, eval_output_noise="clean"))
    result = dict(**ctx.receipt(), evidence=evidence,
                  cost=dict(completed_steps=game.policy.completed_steps, optimizer_updates=counts,
                            adapter_loop_seconds=time.monotonic() - start))
    atomic_json(output / "adapter-receipt.json", result)
    return result
