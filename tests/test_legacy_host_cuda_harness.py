"""CPU reference parity, and CUDA wrappers for the repaired-host smoke.

The CPU check runs each host against the pre-change fixture and RNG code in
this process. It does not store a hash. CUDA cases skip without a GPU. The
A6000 command does not need pytest:

    python -m benchmarks.locked_shared.cuda_host_smoke
"""

from contextlib import contextmanager
import copy

import pytest
import torch
from torch import nn

from benchmarks.locked_shared import cuda_host_smoke
from benchmarks.locked_shared.hosts import (
    ae_gan_hold, cover_leftover, unipolar, unused_token_hold,
)
from benchmarks.toy100.device import experiment_generator
from benchmarks.toy100.train import train as train_toy100
from benchmarks.toy_suite import _check_toy100_policy
from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy


COVER_FLOAT_KEYS = (
    "content_kept", "cover_weight", "even_norm", "fm_weight", "kappa", "leak_frac",
    "leak_ratio", "odd_norm", "on_content", "on_e", "on_u", "particle_l2",
    "particle_rms", "pole_rel_err_minus", "pole_rel_err_plus", "reg_coeff",
    "same_dir", "u_kept",
)
UNUSED_FLOAT_KEYS = (
    "concept_cos", "concept_delta_norm", "concept_move", "scale0_err",
    "unused_dist", "unused_hold",
)
UNIPOLAR_FLOAT_KEYS = (
    "cover", "off_caption", "neu_hold", "cos_plus", "mag_ratio",
    "half_norm_ratio", "origin_norm", "odd_norm", "even_norm",
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA harness smoke")


class _LegacyCpuRng:
    """The pre-change bracket: torch.get_rng_state / set_rng_state."""

    @property
    def device(self):
        return torch.device("cpu")

    def get_state(self):
        return torch.get_rng_state()

    def set_state(self, state):
        torch.set_rng_state(state)


def _legacy_fourier_init(self, dim, *, n_rand, hidden, seed):
    nn.Module.__init__(self)
    gen = torch.Generator().manual_seed(int(seed) + 17)
    bank = (2.0 * torch.randn(int(n_rand), int(dim), generator=gen, device="cpu")).to(
        torch.get_default_device(),
    )
    self.register_buffer("bank", bank)
    feat = 4 * int(dim) + 2 * int(n_rand)
    self.net = nn.Sequential(
        nn.Linear(feat, int(hidden)),
        nn.LeakyReLU(0.2),
        nn.Linear(int(hidden), int(hidden)),
        nn.LeakyReLU(0.2),
        nn.Linear(int(hidden), 1),
    )


def _legacy_student_init(self):
    nn.Module.__init__(self)
    self.shared = nn.Parameter(torch.zeros(unused_token_hold.DIM))
    self.slot = nn.Parameter(torch.zeros(unused_token_hold.N_SLOTS, unused_token_hold.DIM))
    self.register_buffer("neu", unused_token_hold.NEU.clone())


@contextmanager
def _legacy_unipolar():
    original = unipolar._placed
    unipolar._placed = lambda reference, like: reference
    try:
        yield
    finally:
        unipolar._placed = original


@contextmanager
def _legacy_ae():
    original = ae_gan_hold.experiment_generator
    ae_gan_hold.experiment_generator = lambda: _LegacyCpuRng()
    try:
        yield
    finally:
        ae_gan_hold.experiment_generator = original


@contextmanager
def _legacy_cover():
    original = cover_leftover._FourierCritic.__init__
    cover_leftover._FourierCritic.__init__ = _legacy_fourier_init
    try:
        yield
    finally:
        cover_leftover._FourierCritic.__init__ = original


@contextmanager
def _legacy_unused():
    original_init = unused_token_hold.SharedSlotStudent.__init__
    original_placed = unused_token_hold._placed
    unused_token_hold.SharedSlotStudent.__init__ = _legacy_student_init
    unused_token_hold._placed = lambda reference, like: reference
    try:
        yield
    finally:
        unused_token_hold.SharedSlotStudent.__init__ = original_init
        unused_token_hold._placed = original_placed


def _capture(run, keys):
    torch.manual_seed(0)
    row = run()
    values = torch.tensor([float(row[key]) for key in keys], dtype=torch.float64)
    return values, torch.get_rng_state().clone()


def _matches_legacy(run, keys, legacy):
    current, current_rng = _capture(run, keys)
    with legacy:
        previous, previous_rng = _capture(run, keys)
    assert torch.equal(current, previous)
    assert torch.equal(current_rng, previous_rng)


def test_cpu_two_step_streams_match_the_prechange_reference():
    _matches_legacy(
        lambda: unipolar.run_arm("locked_rpgan", steps=2, seed=0),
        UNIPOLAR_FLOAT_KEYS, _legacy_unipolar(),
    )
    _matches_legacy(
        lambda: ae_gan_hold.train(ae_gan_hold.HoldConfig(
            name="parity", steps=2, batch=16, n_particles=8,
        )),
        ("recon_mse", "hold", "init_recon_mse"), _legacy_ae(),
    )
    _matches_legacy(
        lambda: cover_leftover.fit_cover_leftover(cover_leftover.CoverRecipe(steps=2)),
        COVER_FLOAT_KEYS, _legacy_cover(),
    )
    bank_now = cover_leftover._FourierCritic(4, n_rand=16, hidden=8, seed=3).bank.detach().cpu()
    with _legacy_cover():
        bank_then = cover_leftover._FourierCritic(4, n_rand=16, hidden=8, seed=3).bank.detach().cpu()
    assert torch.equal(bank_now, bank_then)
    _matches_legacy(
        lambda: unused_token_hold.train(unused_token_hold.UnusedHoldRecipe(steps=2)),
        UNUSED_FLOAT_KEYS, _legacy_unused(),
    )


def test_cpu_fixtures_are_not_resampled():
    student = unipolar.FreeOriginResidual(unipolar.DIM)
    assert unipolar._placed(unipolar.PLUS, student.odd) is unipolar.PLUS
    holder = unused_token_hold.SharedSlotStudent()
    assert torch.equal(holder.neu, unused_token_hold.NEU)
    assert holder.neu.data_ptr() != unused_token_hold.NEU.data_ptr()
    assert unused_token_hold._placed(
        unused_token_hold.CONCEPT_DIR, holder.shared,
    ) is unused_token_hold.CONCEPT_DIR


def test_ae_evaluation_leaves_the_training_rng_untouched():
    torch.manual_seed(4)
    cfg = ae_gan_hold.HoldConfig(name="rng", steps=1, batch=8, n_particles=4)
    recipe = ae_gan_hold.make_recipe(cfg)
    prior = recipe.make_prior()
    encoder, decoder = ae_gan_hold.MLP(2, 4), ae_gan_hold.MLP(2, 2)
    before = experiment_generator().get_state().clone()
    ae_gan_hold.evaluate(encoder, decoder, prior, recipe)
    assert torch.equal(before, experiment_generator().get_state())
    expected = torch.randn(5)
    experiment_generator().set_state(before)
    assert torch.equal(expected, torch.randn(5))


def _tiny_grid(tmp_path):
    config = dict(
        problem="grid100", seed=31, steps=2, device="cpu",
        z_dim=2, num_particles=8, batch_size=4, d_hidden=4, n_hidden=1,
        fourier=0, eval_samples=16, snapshot_samples=4, eval_interval=2,
        snapshot_interval=2, log_interval=2, threads=1,
        output_noise_std=0.029, output_noise_learnable=True,
        input_noise_std=0.5, toy100_model="affine_square_v1",
    )
    directory = tmp_path / "grid"
    summary = train_toy100(config, directory)
    resolved = summary["config"]
    policy = declared_model_policy(config)
    return directory, summary, resolved, policy


def test_replay_rejects_tampered_seed_device_config_or_noise(tmp_path):
    directory, summary, resolved, policy = _tiny_grid(tmp_path)
    assert _check_toy100_policy(directory, summary, resolved, policy)

    seeded = dict(resolved)
    seeded["seed"] = resolved["seed"] + 1
    with pytest.raises(ValueError, match="seeded affine initialization differs"):
        _check_toy100_policy(directory, summary, seeded, policy)

    moved = dict(resolved)
    moved["device"] = "cuda"
    with pytest.raises(ValueError, match="recorded device|seeded affine initialization differs"):
        _check_toy100_policy(directory, summary, moved, policy)

    resized = dict(resolved)
    resized["num_particles"] = resolved["num_particles"] + 8
    with pytest.raises(ValueError, match="model policy receipt differs"):
        _check_toy100_policy(directory, summary, resized, policy)

    missing_learned = copy.deepcopy(summary)
    missing_learned.pop("learnable_output_noise")
    with pytest.raises(ValueError, match="learned output-scale receipt is absent"):
        _check_toy100_policy(directory, missing_learned, resolved, policy)

    tampered_noise = copy.deepcopy(summary)
    tampered_noise["learnable_output_noise"]["parameter"] = "not-the-generator-scale"
    with pytest.raises(ValueError, match="learned output scale is not bound"):
        _check_toy100_policy(directory, tampered_noise, resolved, policy)

    isolated = dict(resolved)
    isolated["output_noise_rng"] = "isolated"
    with pytest.raises(ValueError, match="model policy receipt differs|isolated output-noise"):
        _check_toy100_policy(directory, summary, isolated, policy)


def test_candidate_episode_rejects_a_missing_noise_receipt(tmp_path, monkeypatch):
    from test_toy_suite import _write_candidate_episode
    import benchmarks.toy_suite as toy_suite

    monkeypatch.setattr(toy_suite, "test_verdict", lambda spec, result: dict(
        status="PASS", passed=True, convergence=dict(passing_suffix=5),
    ))
    name = _write_candidate_episode(
        tmp_path / "missing", lambda record: record.pop("noise_receipt"),
    )
    grade = toy_suite._episode_rows(tmp_path / "missing", (name,), candidate=True)
    assert grade["status"] == "INVALID"
    assert "noise receipt is absent" in grade["reason"]


def test_repaired_hosts_and_replay_declarations_pin_current_sources():
    import hashlib
    import json
    from pathlib import Path
    from pr_integration_compat import before_pr_integration

    root = Path(__file__).resolve().parents[1]
    cards = [*root.glob("configs/forge/tasks/*.json"),
             *root.glob("configs/forge/task-variants/*/*.json")]
    checked = inherited = 0
    for path in cards:
        task = json.loads(path.read_text())
        # Verify the new exact-parent source-only layer before inspecting the
        # historical declaration. Keep current task bytes for the live hash checks.
        previous = before_pr_integration(root, task, str(path.relative_to(root)))
        refreshed = set()
        if previous is not task:
            layer = task["evaluation"]["evaluator_revision"]
            refreshed.update(layer["previous_source_sha256"])
            refreshed.update(layer.get("added_source_members", ()))
        revision = previous["evaluation"].get("evaluator_revision") or {}
        if revision.get("id") == "guarded-gram-default-receipt-source-binding-v1":
            # One exact source-only layer retains the original PR399 question;
            # include its rebound source in the same strict live-byte check.
            assert revision["previous_task_commit"] == "90ae663551dd0fa35749d56e1859f276a4d09587"
            assert revision["previous_task_path"] == str(path.relative_to(root))
            assert set(revision["previous_source_sha256"]) == {"experiments/forge/behavior_adapters.py"}
            refreshed.update(revision["previous_source_sha256"])
            revision = revision["previous_revision"]
        if revision.get("id") not in {
            "pr399-host-device-replay-source-only-v1",
            "tier2-grid-triad-clean-sampling-calibration-v1",
        }:
            continue
        changed = set(revision["previous_source_sha256"]) | set(revision["added_source_members"]) | refreshed
        for name in changed:
            digest = task["evaluation"]["sources"][name]
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, (path, name)
        if task.get("task_cohort") in {"policy_cloud_coverage_v1", "policy_cloud_coverage_as_served_v1"}:
            # Scoped coverage variants (both scoring laws) inherit their counted parent's revision.
            assert revision == task["execution"]["policy_parent_definition"]["evaluation"]["evaluator_revision"]
            inherited += 1
            continue
        checked += 1
    assert checked == 27 and inherited == 10 + 6


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize("host", cuda_host_smoke.HOSTS)
def test_cuda_two_finite_updates_and_noise_receipt(host):
    cuda_host_smoke.two_finite_updates(host)


@pytest.mark.cuda
@CUDA
def test_cuda_fresh_process_init_and_draw_match():
    cuda_host_smoke.fresh_process_reproducible()


@pytest.mark.cuda
@CUDA
def test_cuda_save_restore_repeats_the_next_update():
    cuda_host_smoke.save_restore_next_update()


@pytest.mark.cuda
@CUDA
def test_cuda_evaluation_leaves_the_training_rng_untouched():
    cuda_host_smoke.evaluation_leaves_training_rng()


@pytest.mark.cuda
@CUDA
def test_cuda_fourier_and_unused_fixtures_keep_cpu_values():
    cuda_host_smoke.fixtures_keep_cpu_values()
