"""Unit, checkpoint-continuation, clock-free, and semantic control tests for Track 1B (t1_coupling)."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from experiments.forge import api as forge_api
from particlegan import (
    GANTrainer,
    InputNoise,
    ParticlePrior,
    T1CouplingCriticAdam,
    T1CouplingCriticPenalty,
    T1CouplingGANLoss,
    T1CouplingLoss,
    bind_t1_coupling_record,
    get_recipe,
    init,
)
from particlegan.t1_coupling import (
    compute_chart_valley_gate,
    deconflicted_chart_routing,
    update_streaming_charts,
)


class _ToyGen(nn.Module):
    def __init__(self, z_dim: int = 4, hidden: int = 16, out_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _ToyDisc(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _four_mode_batch(batch_size: int, rng: torch.Generator) -> torch.Tensor:
    centers = torch.tensor([[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]], dtype=torch.float32)
    idx = torch.randint(0, 4, (batch_size,), generator=rng)
    noise = 0.08 * torch.randn((batch_size, 2), generator=rng)
    return centers[idx] + noise


def test_zero_rng_perturbation_and_multiscale_feydy_zero_bias():
    torch.manual_seed(12345)
    rng_data = torch.Generator().manual_seed(77)
    centers = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])
    widths = torch.tensor([0.07, 0.30])
    comp = torch.randint(0, 2, (32,), generator=rng_data)
    y = centers[comp] + widths[comp].unsqueeze(1) * torch.randn((32, 2), generator=rng_data)
    x = y.clone().requires_grad_(True)
    logits = torch.zeros(32)

    op = T1CouplingLoss(
        weight=1.0,
        rho=3.0,
        eps=0.1,
        iters=24,
        self_coupling="feydy",
        relativistic_weight=0.0,
        slack_gain=0.0,
        tau=0.0,
        multiscale_horizon=True,
        chart_weight=0.0,
        support_completion=False,
    )
    before_rng = torch.get_rng_state()
    loss, _ = op(x, y, logits, logits, symmetric=True)
    after_rng = torch.get_rng_state()
    assert torch.equal(before_rng, after_rng), "T1CouplingLoss must not consume global RNG"
    assert abs(float(loss.item())) < 1e-5, f"Expected exact zero Feydy self-coupling bias at X==Y, got {loss.item()}"
    loss.backward()
    assert x.grad is not None and float(x.grad.abs().max().item()) < 1e-4


def test_streaming_charts_break_batch_horizon_and_avoid_ghost_barycenters():
    rng = torch.Generator().manual_seed(9)
    # 36 grid modes in [-3, 3]^2, sampled with smaller batch B_r = 16 < M = 36!
    coords = torch.linspace(-2.5, 2.5, 6)
    gx, gy = torch.meshgrid(coords, coords, indexing="ij")
    modes = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # [36, 2]

    c_sum = c_cnt = c_sq = c_fm = None
    phase, revs = 0.0, 0
    for step_idx in range(18):
        # Cycle deterministically + random noise so all 36 modes appear across streaming batches of 16
        idx = (torch.arange(16) + 16 * step_idx) % 36
        y_batch = modes[idx] + 0.02 * torch.randn((16, 2), generator=rng)
        charts, w_real, radii, b_cnt, c_sum, c_cnt, c_sq, c_fm, phase, revs = update_streaming_charts(
            y_batch, c_sum, c_cnt, c_sq, c_fm, phase, revs, max_charts=128
        )

    assert charts.shape[0] >= 36, f"Expected chart bank K >= 36 > B_r=16, got K={charts.shape[0]}"
    d_mode_to_chart = torch.cdist(modes, charts).amin(dim=1)
    assert float(d_mode_to_chart.max().item()) < 0.12, "Every grid mode must be represented in persistent chart bank"

    ell_c = torch.tensor(0.03)
    w_val = compute_chart_valley_gate(y_batch, charts, ell_c, rho=3.0)
    assert w_val > 0.8, f"Chart valley gate must detect valleys even when B_r=16 < M=36, got {w_val}"

    # Settled in-mode samples suffer ZERO pull; valley stragglers commit to ONE chart (never 0.5 midpoint).
    x_settled = modes[:16].clone()
    _, target_halos, route_w_settled, _, _ = deconflicted_chart_routing(
        x_settled, charts, w_real, radii, b_cnt, w_real.clone(), ell_c, w_valley=w_val, halo_mult=2.1, allow_update=False
    )
    assert float(route_w_settled.max().item()) == 0.0

    # Valley stragglers placed at midpoints between grid modes
    x_valley = 0.5 * (modes[:16] + modes[1:17])
    target_c, _, route_w_val, _, _ = deconflicted_chart_routing(
        x_valley, charts, w_real, radii, b_cnt, w_real.clone(), ell_c, w_valley=w_val, halo_mult=2.1, allow_update=False
    )
    assert float(route_w_val.mean().item()) > 0.5
    # Each target_c row must lie on an actual chart (near a true mode), never in the empty valley!
    dist_target_to_true = torch.cdist(target_c, modes).amin(dim=1)
    assert float(dist_target_to_true.max().item()) < 0.12


def test_exchangeable_conditional_drift_deconfliction_unbiased_and_accurate():
    rng = torch.Generator().manual_seed(42)
    b_n = 32
    c_cont = torch.randn((b_n, 2), generator=rng)
    cond_disc = torch.zeros((b_n, 1), dtype=torch.int64)
    # Strong linear drift B = [[2.5, -1.0], [0.5, 2.0]] + small innovation sigma = 0.10
    b_true = torch.tensor([[2.5, -1.0], [0.5, 2.0]])
    op_deconf = T1CouplingLoss(weight=1.0, cond_drift_deconflict=True)

    # Check finite-sample swap unbiasedness across 64 random pair swaps at q == p
    grad_accum = torch.zeros(2)
    y_base = c_cont @ b_true + 0.10 * torch.randn((b_n, 2), generator=rng)
    x_base = c_cont @ b_true + 0.10 * torch.randn((b_n, 2), generator=rng)
    for _ in range(64):
        swap = (torch.rand((b_n, 1), generator=rng) < 0.5)
        x_s = torch.where(swap, y_base, x_base)
        y_s = torch.where(swap, x_base, y_base)
        theta = torch.zeros(2, requires_grad=True)
        loss = op_deconf.conditional_fiber_transport_loss(
            x_s + theta, y_s, c_cont, cond_disc=cond_disc, mode="diag_imq_cmmd"
        )
        loss.backward()
        grad_accum += theta.grad.detach()
    mean_grad = grad_accum / 64.0
    assert float(mean_grad.norm().item()) < 2e-3, f"Expected zero swap-averaged gradient bias, got {mean_grad}"

    # Verify drift-deconflicted loss strictly prefers true covariance sigma=0.10 over inflated sigma=0.45
    y_eval = c_cont @ b_true + 0.10 * torch.randn((b_n, 2), generator=rng)
    eps_draw = torch.randn((b_n, 2), generator=rng)
    loss_true = op_deconf.conditional_fiber_transport_loss(
        c_cont @ b_true + 0.10 * eps_draw, y_eval, c_cont, cond_disc=cond_disc, mode="diag_imq_cmmd"
    )
    loss_inflated = op_deconf.conditional_fiber_transport_loss(
        c_cont @ b_true + 0.45 * eps_draw, y_eval, c_cont, cond_disc=cond_disc, mode="diag_imq_cmmd"
    )
    assert float(loss_true.item()) < float(loss_inflated.item())


def test_clock_free_parity_across_horizon_and_eval_cadence():
    def _build_and_run(total_steps: int, eval_every: int):
        recipe = get_recipe("t1_coupling", z_dim=4, num_particles=24, batch_size=16, total_steps=total_steps)
        g = _ToyGen(z_dim=4, hidden=16, out_dim=2)
        d = _ToyDisc(in_dim=2, hidden=16)
        init.deterministic_orthogonal_(g, seed=101)
        init.deterministic_orthogonal_(d, seed=102)
        prior = ParticlePrior(24, 4, generator=torch.Generator().manual_seed(103))
        init.deterministic_orthogonal_(prior, seed=104)
        trainer = GANTrainer(recipe, g, d, prior=prior, seed=0)
        data_rng = torch.Generator().manual_seed(555)
        for step in range(1, 9):
            real = _four_mode_batch(16, data_rng)
            trainer.step(real)
            if eval_every > 0 and step % eval_every == 0:
                _ = trainer.sample(32)
        return trainer

    t_short = _build_and_run(total_steps=50, eval_every=0)
    t_long = _build_and_run(total_steps=5000, eval_every=2)
    for p1, p2 in zip(t_short.G.parameters(), t_long.G.parameters()):
        assert torch.equal(p1, p2), "Generator trajectory depended on total_steps horizon or sample() cadence!"
    for p1, p2 in zip(t_short.D.parameters(), t_long.D.parameters()):
        assert torch.equal(p1, p2), "Critic trajectory depended on total_steps horizon or sample() cadence!"
    assert torch.equal(t_short.prior.z, t_long.prior.z)


def test_fresh_process_checkpoint_continuation_and_tamper_rejection(tmp_path: Path):
    ckpt_path = tmp_path / "trainer_mid.pt"
    out_cont_path = tmp_path / "continued_state.pt"

    recipe = get_recipe("t1_coupling", z_dim=4, num_particles=24, batch_size=16, total_steps=100)
    g = _ToyGen(z_dim=4, hidden=16, out_dim=2)
    d = _ToyDisc(in_dim=2, hidden=16)
    init.deterministic_orthogonal_(g, seed=201)
    init.deterministic_orthogonal_(d, seed=202)
    prior = ParticlePrior(24, 4, generator=torch.Generator().manual_seed(203))
    init.deterministic_orthogonal_(prior, seed=204)
    trainer = GANTrainer(recipe, g, d, prior=prior, seed=0)

    data_rng = torch.Generator().manual_seed(888)
    batches = [_four_mode_batch(16, data_rng) for _ in range(10)]
    batches_path = tmp_path / "batches.pt"
    torch.save(batches, batches_path)

    for b in batches[:5]:
        trainer.step(b)
    assert trainer.opt_d.record.chart_initialized is True
    assert trainer.opt_d.record.t1c_thermal_initialized is True
    torch.save(trainer.state_dict(), ckpt_path)

    # Continue uninterrupted in current process
    for b in batches[5:]:
        trainer.step(b)

    # Continue from step 5 in a genuinely fresh Python process using weights_only=True
    script = f"""
import torch
from particlegan import GANTrainer, ParticlePrior, get_recipe, init
from tests.test_t1_coupling import _ToyGen, _ToyDisc

recipe = get_recipe("t1_coupling", z_dim=4, num_particles=24, batch_size=16, total_steps=100)
g = _ToyGen(z_dim=4, hidden=16, out_dim=2)
d = _ToyDisc(in_dim=2, hidden=16)
init.deterministic_orthogonal_(g, seed=999)
init.deterministic_orthogonal_(d, seed=998)
prior = ParticlePrior(24, 4, generator=torch.Generator().manual_seed(997))
trainer2 = GANTrainer(recipe, g, d, prior=prior, seed=77)
state = torch.load({str(ckpt_path)!r}, weights_only=True)
trainer2.load_state_dict(state)
batches = torch.load({str(batches_path)!r}, weights_only=True)
for b in batches[5:]:
    trainer2.step(b)
torch.save({{
    "g": [p.detach() for p in trainer2.G.parameters()],
    "d": [p.detach() for p in trainer2.D.parameters()],
    "z": trainer2.prior.z.detach(),
    "record": trainer2.opt_d.record.state_dict(),
    "sample": trainer2.sample(16),
}}, {str(out_cont_path)!r})
"""
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"Fresh-process continuation failed:\n{proc.stderr}"

    resumed = torch.load(out_cont_path, weights_only=True)
    for p_ref, p_res in zip(trainer.G.parameters(), resumed["g"]):
        assert torch.equal(p_ref.detach(), p_res)
    for p_ref, p_res in zip(trainer.D.parameters(), resumed["d"]):
        assert torch.equal(p_ref.detach(), p_res)
    assert torch.equal(trainer.prior.z.detach(), resumed["z"])
    assert torch.equal(trainer.sample(16), resumed["sample"])
    assert torch.equal(trainer.opt_d.record.chart_sum, resumed["record"]["t1_coupling"]["chart_sum"])
    assert torch.equal(trainer.opt_d.record.chart_fake_mass, resumed["record"]["t1_coupling"]["chart_fake_mass"])

    # Tamper rejection on observer & formulation critic records
    bad_state = deepcopy(torch.load(ckpt_path, weights_only=True))
    bad_state["optimizers"][1]["regularizer"]["record"]["t1_coupling"]["schema"] = 99
    with pytest.raises(ValueError, match="schema"):
        trainer.opt_d.load_state_dict(bad_state["optimizers"][1])
    with pytest.raises(ValueError, match="invalid checkpoint optimizer state"):
        trainer.load_state_dict(bad_state)

    # Also test T1CouplingCriticAdam (formulation family) save/load & tamper rejection
    form_recipe = get_recipe(
        "ka2",
        critic_formulation="t1_coupling",
        optimizer_family="formulation",
        reg_anchor_weight=0.0,
        d_guard_ratio=0.0,
        t1c_weight=1.0,
        t1c_chart_weight=1.0,
    )
    d_form = _ToyDisc()
    opt_form = form_recipe.make_critic_optimizer(d_form)
    assert isinstance(opt_form, T1CouplingCriticAdam)
    pen_form = form_recipe.make_critic_penalty(opt_form, collect_stats=True)
    assert isinstance(pen_form, T1CouplingCriticPenalty)
    p_val = pen_form(d_form, batches[0], batches[1])
    assert pen_form.last_stats["applied"] is True
    p_val.backward()
    opt_form.step()
    saved_form = opt_form.state_dict()
    opt_form2 = form_recipe.make_critic_optimizer(_ToyDisc())
    opt_form2.load_state_dict(saved_form)
    assert opt_form2.record.observed_steps == 1
    tampered_form = deepcopy(saved_form)
    tampered_form["regularizer"]["record"]["formulation"] = "ka2"
    with pytest.raises(ValueError, match="t1_coupling"):
        opt_form2.load_state_dict(tampered_form)


def test_deliberate_damage_control_and_forge_extension_bindings():
    # Forge ExtensionSpec registry check on both execution paths
    for execution_path in ("public_trainer", "public_components"):
        ctx = forge_api.FormulationContext(
            recipe_preset="t1_coupling",
            execution_path=execution_path,
            extensions={"t1c_chart_weight": 1.25, "t1c_grad_deconflict": 0.4},
        )
        assert ctx.recipe.t1c_chart_weight == 1.25
        assert ctx.recipe.t1c_grad_deconflict == 0.4
        receipt = ctx.receipt()
        defined = {item["public_definition"] for item in receipt["api_changes"]}
        assert "particlegan.Recipe.t1c_chart_weight" in defined
        assert "particlegan.Recipe.t1c_grad_deconflict" in defined

    # Semantic negative control: deliberate damage to G increases T1CouplingGANLoss.g_loss
    recipe = get_recipe("t1_coupling", z_dim=4, num_particles=32, batch_size=32, total_steps=60)
    g = _ToyGen(z_dim=4, hidden=32, out_dim=2)
    d = _ToyDisc(in_dim=2, hidden=32)
    init.deterministic_orthogonal_(g, seed=11)
    init.deterministic_orthogonal_(d, seed=12)
    prior = ParticlePrior(32, 4, generator=torch.Generator().manual_seed(13))
    trainer = GANTrainer(recipe, g, d, prior=prior, seed=0)
    assert isinstance(trainer.loss, T1CouplingGANLoss)

    rng = torch.Generator().manual_seed(33)
    for _ in range(40):
        trainer.step(_four_mode_batch(32, rng))

    eval_real = _four_mode_batch(32, rng)
    with torch.no_grad():
        good_samples = trainer.sample(64)
        good_d_real = trainer.D(eval_real)
        good_d_fake = trainer.D(good_samples[:32])
        good_loss = float(trainer.loss.g_loss(good_d_fake, good_d_real).item())

        # Deliberately corrupt generator parameters (collapse to far outlier)
        for p in trainer.G.parameters():
            p.zero_()
        trainer.G.net[-1].bias.fill_(25.0)
        bad_samples = trainer.sample(64)
        bad_d_fake = trainer.D(bad_samples[:32])
        bad_loss = float(trainer.loss.g_loss(bad_d_fake, good_d_real).item())

    assert bad_loss > good_loss + 1.0, f"Deliberate generator damage must worsen g_loss ({bad_loss} vs {good_loss})"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA determinism check")
def test_cuda_strict_deterministic_algorithms_highres_conv():
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    prev_det = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        dev = torch.device("cuda:0")
        recipe = get_recipe("t1_coupling", z_dim=8, num_particles=32, batch_size=16, total_steps=20)
        g = nn.Sequential(
            nn.Linear(8, 64 * 4 * 4),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, 4, 4, 0),
            nn.Tanh(),
        ).to(dev)
        d = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 4, 0),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 1),
        ).to(dev)
        prior = ParticlePrior(32, 8, device=dev, generator=torch.Generator(device=dev).manual_seed(5))
        trainer = GANTrainer(recipe, g, d, prior=prior, seed=0)
        rng = torch.Generator(device=dev).manual_seed(7)
        for _ in range(4):
            real = torch.randn((16, 3, 32, 32), device=dev, generator=rng)
            out = trainer.step(real, collect_stats=True)
            assert bool(torch.isfinite(out["loss_g"])) and bool(torch.isfinite(out["loss_d"]))
    finally:
        torch.use_deterministic_algorithms(prev_det)


def test_disconnected_h0_screen_skips_simplicial_chord_interpolation():
    loss_op = T1CouplingLoss(weight=1.0)
    rng = torch.Generator().manual_seed(17)
    z_tab = torch.randn(16, 4, generator=rng, requires_grad=True)
    g_net = _ToyGen(z_dim=4, hidden=16, out_dim=2)
    calls = {"n": 0}

    def counted_gen(z_q: torch.Tensor) -> torch.Tensor:
        calls["n"] += 1
        return g_net(z_q)

    # Disconnected 4-mode cloud (H_0 bottleneck persistence >> 10 * q90_nn)
    y_disc = _four_mode_batch(64, rng)
    s_disc = loss_op.simplicial_manifold_loss(z_tab, counted_gen, y_disc, chart_weight=4.0)
    assert float(s_disc.item()) == 0.0
    assert calls["n"] == 0, "Disconnected target must not invoke generator on cross-valley chords"

    # Connected 1D circular arc manifold (g_conn > 0.5)
    theta = torch.linspace(-0.5, 3.0, 64)
    y_arc = torch.stack([2.0 * torch.cos(theta), 2.0 * torch.sin(theta)], dim=-1)
    y_arc = y_arc + 0.02 * torch.randn(y_arc.shape, generator=rng)
    s_arc = loss_op.simplicial_manifold_loss(z_tab, counted_gen, y_arc, chart_weight=4.0)
    assert float(s_arc.item()) > 0.0
    assert calls["n"] >= 2


def _unequal_width_8mode_batch(batch_size: int, step_idx: int, rng: torch.Generator, shift: float = 0.0) -> torch.Tensor:
    """12-mode unequal-width mixture cycled 6 modes per batch so B_r=16 has both multi-sample patches and 6 absent charts."""
    angles = torch.linspace(0.0, 2.0 * 3.14159265 * (11.0 / 12.0), 12)
    centers = 0.75 * torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1) + shift
    # Strong width heterogeneity across modes (0.006 vs 0.085) so patch-smoothed g_width > 0 activates!
    widths = torch.tensor([0.006, 0.006, 0.007, 0.008, 0.008, 0.010, 0.060, 0.070, 0.075, 0.080, 0.082, 0.085])
    start = (3 * step_idx) % 12
    active_modes = (start + torch.arange(6)) % 12
    pick = active_modes[torch.randint(0, 6, (batch_size,), generator=rng)]
    return centers[pick] + widths[pick].unsqueeze(1) * torch.randn((batch_size, 2), generator=rng)


def _connected_arc_batch(batch_size: int, rng: torch.Generator) -> torch.Tensor:
    theta = -1.2 + 2.4 * torch.rand((batch_size,), generator=rng)
    arc = 2.0 * torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    return arc + 0.015 * torch.randn((batch_size, 2), generator=rng)


def test_fresh_process_continuation_across_all_active_mechanism_boundaries(tmp_path: Path):
    """Verify bit-identical fresh-process continuation across active cooling, shock reopening,
    K>4 streaming charts + absent_shield + revival + conformal g_width + grad_deconflict, and
    zero-sigma ParticlePrior connected-manifold simplicial loss."""
    # Part A: Active thermal cooling (< 1.0) -> checkpoint -> abrupt target shift shock reopening (== 1.0),
    # simultaneous with K >= 8 chart atlas, absent_shield, chart revival, conformal g_width > 0, and grad_deconflict.
    ckpt_a = tmp_path / "active_boundary_a.pt"
    out_a = tmp_path / "resumed_a.pt"
    batches_a_path = tmp_path / "batches_a.pt"

    rng_a = torch.Generator().manual_seed(1234)
    # Steps 0..5: 12-mode unequal-width mixture; Step 6..8: abrupt +25.0 distribution shift triggers shock reopen!
    batches_a = [_unequal_width_8mode_batch(16, i, rng_a, shift=0.0) for i in range(6)] + [
        _unequal_width_8mode_batch(16, i + 6, rng_a, shift=25.0) for i in range(3)
    ]
    torch.save(batches_a, batches_a_path)

    # Set rhbi_thermal_tau=2.80 so step 5 cools _thermal_scale ~ 0.285 < 0.95 at the checkpoint boundary,
    # and the +25.0 shift on steps 6..8 triggers shock reopening back to _thermal_scale == 1.0!
    recipe_kwargs_a = dict(
        z_dim=4,
        num_particles=24,
        batch_size=16,
        total_steps=60,
        rhbi_thermal_tau=2.80,
        t1c_chart_weight=0.85,
        t1c_multiscale_horizon=True,
        t1c_grad_deconflict=0.30,
    )
    recipe_a = get_recipe("t1_coupling", **recipe_kwargs_a)
    g_a = _ToyGen(z_dim=4, hidden=16, out_dim=2)
    d_a = _ToyDisc(in_dim=2, hidden=16)
    init.deterministic_orthogonal_(g_a, seed=301)
    init.deterministic_orthogonal_(d_a, seed=302)
    prior_a = recipe_a.make_prior(generator=torch.Generator().manual_seed(303))
    init.deterministic_orthogonal_(prior_a, seed=304)
    trainer_a = GANTrainer(recipe_a, g_a, d_a, prior=prior_a, seed=0)

    # Verify g_width > 0 actually activates on batches_a[0]
    ell_x_ms, ell_y_ms, _ = trainer_a.loss.t1c._conformal_local_horizons(
        batches_a[0], batches_a[0], torch.tensor(0.15)
    )
    assert not torch.allclose(ell_y_ms, torch.full_like(ell_y_ms, 0.15)), "Expected active g_width > 0 horizon"

    for b in batches_a[:6]:
        trainer_a.step(b)

    rec_a = trainer_a.opt_d.record
    assert rec_a.chart_initialized is True and rec_a.chart_sum.shape[0] >= 8
    assert rec_a.t1c_het_initialized is True
    assert 0.0 < rec_a._thermal_scale < 0.95, f"Expected active thermal cooling < 0.95 at boundary, got {rec_a._thermal_scale}"
    # Force one chart count just above revival border so post-checkpoint step also exercises revival/rebalance
    rec_a.chart_count[0] = 0.009 * rec_a.chart_count.mean()
    torch.save(trainer_a.state_dict(), ckpt_a)

    # Uninterrupted continuation across the +25.0 shift shock reopening boundary
    scales_post = []
    for b in batches_a[6:]:
        trainer_a.step(b)
        scales_post.append(float(trainer_a.opt_d.record._thermal_scale))
    assert max(scales_post) >= 0.99, f"Expected post-shift shock reopening to ~1.0, got {scales_post}"

    # Part B: Zero-sigma ParticlePrior on connected circular arc (active simplicial_manifold_loss g_conn > 1e-4)
    ckpt_b = tmp_path / "active_boundary_b.pt"
    out_b = tmp_path / "resumed_b.pt"
    batches_b_path = tmp_path / "batches_b.pt"
    rng_b = torch.Generator().manual_seed(5678)
    batches_b = [_connected_arc_batch(24, rng_b) for _ in range(6)]
    torch.save(batches_b, batches_b_path)

    recipe_kwargs_b = dict(
        z_dim=4,
        num_particles=24,
        batch_size=24,
        total_steps=40,
        prior_kind="particles",
        sigma_rel=0.0,
        standardize=False,
        rhbi_simplex_weight=0.35,
        rhbi_simplex_chart_weight=4.0,
    )
    recipe_b = get_recipe("t1_coupling", **recipe_kwargs_b)
    g_b = _ToyGen(z_dim=4, hidden=16, out_dim=2)
    d_b = _ToyDisc(in_dim=2, hidden=16)
    init.deterministic_orthogonal_(g_b, seed=401)
    init.deterministic_orthogonal_(d_b, seed=402)
    prior_b = ParticlePrior(24, 4, generator=torch.Generator().manual_seed(403))
    init.deterministic_orthogonal_(prior_b, seed=404)
    trainer_b = GANTrainer(recipe_b, g_b, d_b, prior=prior_b, seed=0)

    for b in batches_b[:3]:
        res_b = trainer_b.step(b)
    assert "rhbi_simplex" in res_b and float(res_b["rhbi_simplex"].item()) > 0.0, "Expected active simplicial loss"
    torch.save(trainer_b.state_dict(), ckpt_b)
    for b in batches_b[3:]:
        trainer_b.step(b)

    script = f"""
import torch
from particlegan import GANTrainer, ParticlePrior, get_recipe, init
from tests.test_t1_coupling import _ToyGen, _ToyDisc

# Resume Part A in fresh process
recipe_a = get_recipe("t1_coupling", **{recipe_kwargs_a!r})
g_a = _ToyGen(z_dim=4, hidden=16, out_dim=2)
d_a = _ToyDisc(in_dim=2, hidden=16)
init.deterministic_orthogonal_(g_a, seed=911)
init.deterministic_orthogonal_(d_a, seed=912)
prior_a = recipe_a.make_prior(generator=torch.Generator().manual_seed(913))
tr_a = GANTrainer(recipe_a, g_a, d_a, prior=prior_a, seed=99)
tr_a.load_state_dict(torch.load({str(ckpt_a)!r}, weights_only=True))
for b in torch.load({str(batches_a_path)!r}, weights_only=True)[6:]:
    tr_a.step(b)
torch.save({{
    "g": [p.detach() for p in tr_a.G.parameters()],
    "d": [p.detach() for p in tr_a.D.parameters()],
    "z": tr_a.prior.z.detach(),
    "rec": tr_a.opt_d.record.state_dict(),
    "sample": tr_a.sample(64),
}}, {str(out_a)!r})

# Resume Part B in fresh process
recipe_b = get_recipe("t1_coupling", **{recipe_kwargs_b!r})
g_b = _ToyGen(z_dim=4, hidden=16, out_dim=2)
d_b = _ToyDisc(in_dim=2, hidden=16)
init.deterministic_orthogonal_(g_b, seed=811)
init.deterministic_orthogonal_(d_b, seed=812)
prior_b = ParticlePrior(24, 4, generator=torch.Generator().manual_seed(813))
tr_b = GANTrainer(recipe_b, g_b, d_b, prior=prior_b, seed=88)
tr_b.load_state_dict(torch.load({str(ckpt_b)!r}, weights_only=True))
for b in torch.load({str(batches_b_path)!r}, weights_only=True)[3:]:
    tr_b.step(b)
torch.save({{
    "g": [p.detach() for p in tr_b.G.parameters()],
    "d": [p.detach() for p in tr_b.D.parameters()],
    "z": tr_b.prior.z.detach(),
    "rec": tr_b.opt_d.record.state_dict(),
    "sample": tr_b.sample(64),
}}, {str(out_b)!r})
"""
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"Fresh-process active-boundary continuation failed:\n{proc.stderr}"

    res_a = torch.load(out_a, weights_only=True)
    for p_ref, p_res in zip(trainer_a.G.parameters(), res_a["g"]):
        assert torch.equal(p_ref.detach(), p_res)
    for p_ref, p_res in zip(trainer_a.D.parameters(), res_a["d"]):
        assert torch.equal(p_ref.detach(), p_res)
    assert torch.equal(trainer_a.prior.z.detach(), res_a["z"])
    assert torch.equal(trainer_a.sample(64), res_a["sample"])
    assert torch.equal(trainer_a.opt_d.record.chart_sum, res_a["rec"]["t1_coupling"]["chart_sum"])
    assert trainer_a.opt_d.record._thermal_scale == res_a["rec"]["t1_coupling"]["thermal_scale"]

    res_b = torch.load(out_b, weights_only=True)
    for p_ref, p_res in zip(trainer_b.G.parameters(), res_b["g"]):
        assert torch.equal(p_ref.detach(), p_res)
    for p_ref, p_res in zip(trainer_b.D.parameters(), res_b["d"]):
        assert torch.equal(p_ref.detach(), p_res)
    assert torch.equal(trainer_b.prior.z.detach(), res_b["z"])
    assert torch.equal(trainer_b.sample(64), res_b["sample"])


def test_input_noise_joint_bigan_and_fresh_process_step_damp_continuation(tmp_path: Path):
    """Verify InputNoise clean capture, joint_g_loss, active cond_step_damp + step_damp (< 1.0),
    and fresh-process checkpoint + InputNoise RNG stream continuation."""
    recipe_cap = get_recipe("t1_coupling", total_steps=10, batch_size=8, num_particles=32, z_dim=2)

    class _CondJointCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Linear(6, 1)
            self.emb = nn.Embedding(4, 2)

        def forward(self, x: torch.Tensor, z: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
            return self.net(torch.cat([x, z, self.emb(cls)], dim=-1))

    d_cap = _CondJointCritic()
    opt_d_cap = recipe_cap.make_critic_optimizer(d_cap)
    loss_cap = recipe_cap.make_loss()
    bind_t1_coupling_record(opt_d_cap.record, loss_cap)
    assert isinstance(loss_cap, T1CouplingGANLoss)

    # 1. InputNoise records clean x rather than noisy x
    noisy_d = InputNoise(d_cap, std=0.5, generator=torch.Generator().manual_seed(3))
    x_clean = torch.ones(8, 2, requires_grad=True)
    z_clean = torch.zeros(8, 2, requires_grad=True)
    cls_a = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    out = noisy_d(x_clean, z_clean, cls_a).squeeze(-1)
    x_resolved, _, cond_resolved, _, _, _ = loss_cap._resolve_pair_full(out, out.detach())
    assert torch.equal(x_resolved[:, :2], x_clean)
    assert torch.equal(cond_resolved.squeeze(-1), cls_a)

    # 2. joint_g_loss delivers finite nonzero gradients to both G and E streams
    g_lin = nn.Linear(2, 2)
    e_lin = nn.Linear(2, 2)
    z_in = torch.randn(8, 2)
    real_x = torch.randn(8, 2)
    j_loss = loss_cap.joint_g_loss(
        d_cap(g_lin(z_in), z_in, cls_a).squeeze(-1),
        d_cap(real_x, e_lin(real_x), cls_a).squeeze(-1),
    )
    j_loss.backward()
    assert g_lin.weight.grad is not None and float(g_lin.weight.grad.abs().sum()) > 0.0
    assert e_lin.weight.grad is not None and float(e_lin.weight.grad.abs().sum()) > 0.0

    # 3. t1c_cond_ema < 0.028 activates exact generator DualNorm cond_step_damp while leaving D untouched
    r_undamp = get_recipe(
        "t1_coupling",
        constraint_geometry_mode="none",
        rhbi_cond_step_damp=0.0,
        t1c_grad_deconflict=0.0,
        total_steps=4,
        batch_size=8,
        num_particles=8,
        z_dim=2,
    )
    r_damp = get_recipe(
        "t1_coupling",
        constraint_geometry_mode="none",
        rhbi_cond_step_damp=0.68,
        t1c_grad_deconflict=0.0,
        total_steps=4,
        batch_size=8,
        num_particles=8,
        z_dim=2,
    )
    g_u = nn.Linear(2, 2, bias=False)
    init.deterministic_orthogonal_(g_u, seed=1)
    g_d = deepcopy(g_u)
    d_u = nn.Linear(2, 1, bias=False)
    init.deterministic_orthogonal_(d_u, seed=2)
    d_d = deepcopy(d_u)
    opt_g_u, opt_d_u = r_undamp.make_optimizers(g_u, d_u)
    opt_g_d, opt_d_d = r_damp.make_optimizers(g_d, d_d)
    opt_d_d.record.t1c_cond_initialized = True
    opt_d_d.record.t1c_cond_ema = 0.005
    w0 = g_d.weight.detach().clone()
    d0 = d_d.weight.detach().clone()
    (g_u(torch.ones(4, 2)).sum()).backward()
    (g_d(torch.ones(4, 2)).sum()).backward()
    (d_u(torch.ones(4, 2)).sum()).backward()
    (d_d(torch.ones(4, 2)).sum()).backward()
    opt_g_u.step()
    opt_g_d.step()
    opt_d_u.step()
    opt_d_d.step()
    delta_u = float((g_u.weight.detach() - w0).norm())
    delta_d = float((g_d.weight.detach() - w0).norm())
    assert delta_d == pytest.approx(delta_u * (1.0 - 0.68), rel=1e-4)
    assert torch.equal(d_d.weight.detach() - d0, d_u.weight.detach() - d0)

    # 4. Fresh-process continuation across active InputNoise + rhbi_step_damp (< 1.0) + t1c_cond_ema (< 0.028)
    ckpt_c = tmp_path / "active_boundary_c.pt"
    out_c = tmp_path / "resumed_c.pt"
    batches_c_path = tmp_path / "batches_c.pt"
    rng_c = torch.Generator().manual_seed(777)
    batches_c = [_unequal_width_8mode_batch(16, i, rng_c) for i in range(6)]
    torch.save(batches_c, batches_c_path)

    recipe_kwargs_c = dict(
        z_dim=4,
        num_particles=24,
        batch_size=16,
        total_steps=40,
        rhbi_thermal_tau=6.5,
        rhbi_step_damp=0.45,
        rhbi_cond_step_damp=0.68,
        input_noise_std=0.05,
        input_noise_anneal_end=0.8,
    )
    recipe_c = get_recipe("t1_coupling", **recipe_kwargs_c)
    g_c = _ToyGen(z_dim=4, hidden=16, out_dim=2)
    d_c = _ToyDisc(in_dim=2, hidden=16)
    init.deterministic_orthogonal_(g_c, seed=501)
    init.deterministic_orthogonal_(d_c, seed=502)
    prior_c = recipe_c.make_prior(generator=torch.Generator().manual_seed(503))
    init.deterministic_orthogonal_(prior_c, seed=504)
    trainer_c = GANTrainer(recipe_c, g_c, d_c, prior=prior_c, seed=0)

    for b in batches_c[:3]:
        trainer_c.step(b)
    assert trainer_c.opt_d.record._thermal_scale < 0.95
    trainer_c.opt_d.record.t1c_cond_initialized = True
    trainer_c.opt_d.record.t1c_cond_ema = 0.008
    torch.save(trainer_c.state_dict(), ckpt_c)
    for b in batches_c[3:]:
        trainer_c.step(b)

    script_c = f"""
import torch
from particlegan import GANTrainer, get_recipe, init
from tests.test_t1_coupling import _ToyGen, _ToyDisc

recipe_c = get_recipe("t1_coupling", **{recipe_kwargs_c!r})
g_c = _ToyGen(z_dim=4, hidden=16, out_dim=2)
d_c = _ToyDisc(in_dim=2, hidden=16)
init.deterministic_orthogonal_(g_c, seed=711)
init.deterministic_orthogonal_(d_c, seed=712)
prior_c = recipe_c.make_prior(generator=torch.Generator().manual_seed(713))
tr_c = GANTrainer(recipe_c, g_c, d_c, prior=prior_c, seed=55)
tr_c.load_state_dict(torch.load({str(ckpt_c)!r}, weights_only=True))
for b in torch.load({str(batches_c_path)!r}, weights_only=True)[3:]:
    tr_c.step(b)
torch.save({{
    "g": [p.detach() for p in tr_c.G.parameters()],
    "d": [p.detach() for p in tr_c.D.parameters()],
    "z": tr_c.prior.z.detach(),
    "in_rng": tr_c.input_noise_generator.get_state(),
    "rec": tr_c.opt_d.record.state_dict(),
    "sample": tr_c.sample(64),
}}, {str(out_c)!r})
"""
    proc_c = subprocess.run([sys.executable, "-c", script_c], capture_output=True, text=True, check=False)
    assert proc_c.returncode == 0, f"Fresh-process step-damp/InputNoise continuation failed:\n{proc_c.stderr}"

    res_c = torch.load(out_c, weights_only=True)
    for p_ref, p_res in zip(trainer_c.G.parameters(), res_c["g"]):
        assert torch.equal(p_ref.detach(), p_res)
    for p_ref, p_res in zip(trainer_c.D.parameters(), res_c["d"]):
        assert torch.equal(p_ref.detach(), p_res)
    assert torch.equal(trainer_c.prior.z.detach(), res_c["z"])
    assert torch.equal(trainer_c.input_noise_generator.get_state(), res_c["in_rng"])
    assert res_c["rec"]["t1_coupling"]["cond_initialized"] is True
    assert res_c["rec"]["t1_coupling"]["cond_ema"] == pytest.approx(0.008)
    assert torch.equal(trainer_c.sample(64), res_c["sample"])




