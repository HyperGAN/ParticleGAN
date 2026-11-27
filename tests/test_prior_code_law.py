"""Recipe.prior_code_law: the declared fixed-Gaussian latent control (prior ablation 2026-10-25)."""
import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe
from particlegan.particle_prior import MoGParticlePrior, ParticlePrior


def _mog(seed=0, sigma=.025):
    return MoGParticlePrior(16, 3, generator=torch.Generator().manual_seed(seed), sigma=sigma, standardize=False)


def test_default_table_law_is_unchanged():
    prior = _mog()
    assert prior.code_law == "table"
    g = torch.Generator().manual_seed(5)
    codes, idx = prior.sample(8, torch.Generator().manual_seed(5))
    expected_idx = torch.randint(0, 16, (8,), generator=g)
    expected = prior.z[expected_idx] + .025 * torch.randn((8, 3), generator=g)
    assert torch.equal(idx, expected_idx)
    assert torch.equal(codes, expected)


def test_fresh_gaussian_matches_table_index_and_noise_streams():
    table, fresh = _mog(), _mog()
    rows = fresh.z.detach()
    del fresh.z
    fresh.register_buffer("z", rows)
    fresh.set_code_law("fresh_gaussian")
    gi, gn = torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)
    ti, tn = torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)
    codes_t, idx_t = table.sample(32, ti, noise_generator=tn)
    codes_f, idx_f = fresh.sample(32, gi, noise_generator=gn)
    # Same row draws and the same Gaussian stream position: the fresh code is
    # exactly the eps the MoG kernel used, so only the table offset differs.
    assert torch.equal(idx_t, idx_f)
    assert torch.allclose(codes_f, (codes_t - table.z.detach()[idx_t]) / .025, atol=1e-5)
    assert torch.equal(gi.get_state(), ti.get_state()) and torch.equal(gn.get_state(), tn.get_state())


def test_fresh_gaussian_particle_cloud_draws_n01_codes_and_refuses_table_reads():
    prior = ParticlePrior(64, 4, learnable=False, generator=torch.Generator().manual_seed(0))
    prior.set_code_law("fresh_gaussian")
    codes, idx = prior.sample(4096, torch.Generator().manual_seed(3))
    assert codes.shape == (4096, 4) and idx.shape == (4096,)
    assert not any(torch.equal(code, row) for code in codes[:8] for row in prior.z)
    assert abs(codes.mean().item()) < .05 and abs(codes.std().item() - 1) < .05
    with pytest.raises(ValueError, match="not table reads"):
        prior(idx)
    with pytest.raises(ValueError, match="code_law"):
        prior.set_code_law("other")
    with pytest.raises(ValueError, match="frozen location table"):
        ParticlePrior(4, 2).set_code_law("fresh_gaussian")


def test_recipe_declares_and_applies_code_law():
    default = get_recipe("bispf")
    assert default.prior_code_law == "table" and "prior_code_law" not in default.to_dict()
    # Hosts build the initial table through recipe.replace(prior_update="learned");
    # that construction must work (regression: it crashed every Forge host).
    fresh = get_recipe("bispf", prior_update="frozen", prior_code_law="fresh_gaussian")
    table = fresh.replace(prior_update="learned").make_prior()
    with pytest.raises(ValueError, match="requires prior_update='frozen'"):
        fresh.replace(prior_update="learned").apply_prior_policy(table)
    fresh.apply_prior_policy(table)
    assert table.code_law == "fresh_gaussian"
    with pytest.raises(ValueError, match="prior_code_law must be"):
        get_recipe("bispf", prior_update="frozen", prior_code_law="normal")
    recipe = get_recipe("bispf", prior_update="frozen", prior_code_law="fresh_gaussian")
    assert recipe.to_dict()["prior_code_law"] == "fresh_gaussian"
    prior = recipe.replace(prior_update="learned", prior_code_law="table").make_prior()
    recipe.apply_prior_policy(prior)
    assert prior.code_law == "fresh_gaussian" and not prior.z.requires_grad
    with pytest.raises(ValueError, match="contradicts"):
        recipe.replace(prior_code_law="table").apply_prior_policy(prior)


def test_trainer_with_fresh_gaussian_codes_keeps_table_fixed():
    recipe = get_recipe("bispf", prior_update="frozen", prior_code_law="fresh_gaussian",
                        num_particles=32, z_dim=2, batch_size=16)
    torch.manual_seed(0)
    G, D = nn.Linear(2, 2), nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
    trainer = GANTrainer(recipe, G, D, seed=0, max_steps=4)
    before = trainer.prior.z.clone()
    data = torch.randn(64, 2)
    for _ in range(3):
        trainer.step(data[torch.randint(0, 64, (16,))])
    assert torch.equal(trainer.prior.z, before)
    assert trainer.prior.code_law == "fresh_gaussian"
    assert torch.isfinite(trainer.sample(32)).all()


def test_forge_binds_and_blocks_the_code_law():
    from experiments.forge.api import fresh_gaussian_blockers
    from experiments.forge.priors import prior_policy_receipt, validate_prior_policy
    fresh = get_recipe("bispf", prior_update="frozen", prior_code_law="fresh_gaussian")
    mog = {"kind": "mog", "sigma": .025, "standardize": False}
    vector = {"id": "v", "adapter": "transfer_vector", "execution": {"prior": mog}}
    model = {"id": "m", "adapter": "public_api_model_toy", "execution": {"prior": mog}}
    cloud = {"id": "c", "adapter": "transfer_image",
             "execution": {"prior": {"kind": "particle_cloud", "sigma": 0., "standardize": False}}}
    hold = {"id": "h", "adapter": "transfer_behavior", "execution": {"prior": mog, "host": "mode_hold"}}
    assert fresh_gaussian_blockers(vector, fresh) == [] and fresh_gaussian_blockers(hold, fresh) == []
    assert fresh_gaussian_blockers(model, fresh) and fresh_gaussian_blockers(cloud, fresh)
    assert fresh_gaussian_blockers(model, get_recipe("bispf")) == []

    learned = get_recipe("bispf")
    table_prior = learned.make_prior()
    assert "code_law" not in prior_policy_receipt(learned, table_prior)
    prior = fresh.replace(prior_update="learned", prior_code_law="table").make_prior()
    fresh.apply_prior_policy(prior)
    proof = prior_policy_receipt(fresh, prior)
    assert proof["code_law"] == "fresh_gaussian" and proof["actual_code_laws"] == ["fresh_gaussian"]
    task = {"id": "v", "adapter": "transfer_vector",
            "execution": {"prior_contract": "recipe_owned_v1", "prior": mog}}
    recipe_dict = fresh.to_dict()
    assert validate_prior_policy(task, {"prior_policy": proof}, recipe_dict) is None
    tampered = {**proof, "code_law": "table"}
    assert validate_prior_policy(task, {"prior_policy": tampered}, recipe_dict)["status"] == "INVALID"


def test_forge_context_builds_a_fresh_gaussian_prior_for_a_vector_task():
    """Regression: the Forge host path builds the table via replace(prior_update='learned')."""
    import json
    from pathlib import Path
    from experiments.forge.api import task_formulation_context
    root = Path(__file__).resolve().parents[1]
    candidate = json.loads((root / "configs/forge/ideas/bispf-prior-fresh-gaussian-v1.json").read_text())
    task = json.loads((root / "configs/forge/tasks/gaussian1d_smoke.json").read_text())
    context = task_formulation_context(candidate, task, {"seed": 0}, root=root)
    prior = context.build_prior()
    assert prior.code_law == "fresh_gaussian" and not prior.z.requires_grad
    codes, rows = prior.sample(8, torch.Generator().manual_seed(0))
    assert codes.shape == (8, prior.z_dim) and rows.shape == (8,)


def test_manipulated_prior_sample_supports_fresh_gaussian_bases():
    from particlegan.particle_prior import GaussianPrior
    from particlegan.priors import ManipulatedPrior, NormalNoiseManipulator

    pp = ParticlePrior(32, 3, learnable=False, generator=torch.Generator().manual_seed(0))
    pp.set_code_law("fresh_gaussian")
    mp_pp = ManipulatedPrior(pp, [NormalNoiseManipulator(sigma=0.05)])
    codes, idx = mp_pp.sample(16, torch.Generator().manual_seed(1), noise_generator=torch.Generator().manual_seed(2))
    assert codes.shape == (16, 3) and idx.shape == (16,) and torch.isfinite(codes).all()

    mog = _mog()
    rows = mog.z.detach()
    del mog.z
    mog.register_buffer("z", rows)
    mog.set_code_law("fresh_gaussian")
    mp_mog = ManipulatedPrior(mog, [NormalNoiseManipulator(sigma=0.05)])
    codes_m, idx_m = mp_mog.sample(16, torch.Generator().manual_seed(1), noise_generator=torch.Generator().manual_seed(2))
    assert codes_m.shape == (16, 3) and idx_m.shape == (16,) and torch.isfinite(codes_m).all()

    gp = GaussianPrior(3)
    mp_gp = ManipulatedPrior(gp, [NormalNoiseManipulator(sigma=0.05)])
    codes_g, idx_g = mp_gp.sample(16, torch.Generator().manual_seed(1))
    assert codes_g.shape == (16, 3) and idx_g is None and torch.isfinite(codes_g).all()
