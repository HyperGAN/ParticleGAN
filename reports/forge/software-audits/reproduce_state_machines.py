"""CPU software reproductions for implicit-state-machines.md; no research verdicts.

Runtime source: 9fc54fab90a7d6da39db274b651dbf824fc3929a. Follow the isolated
worktree instructions in implicit-state-machines.md to run against that source.
Both defects were fixed in PR #517. Assertions preserve the historical defects
and should fail against fixed source; permanent regressions live in
tests/test_training.py.
"""
import json

import torch
from torch import nn

from particlegan import GANTrainer, get_recipe, init


def build(response="none"):
    recipe = get_recipe(
        "bcap", num_particles=12, z_dim=3, batch_size=6, total_steps=8,
        generator_gradient_response=response,
    )
    generator = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 2))
    critic = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
    init.deterministic_orthogonal_(generator, seed=0)
    init.deterministic_orthogonal_(critic, seed=1)
    prior = init.deterministic_orthogonal_(recipe.make_prior(), seed=0)
    return GANTrainer(recipe, generator, critic, prior=prior, seed=0)


def main():
    torch.set_num_threads(1)
    real = torch.arange(12, dtype=torch.float32).reshape(6, 2) / 12
    for mode in ("none", "coherent_G_mean_v1", "coherent_G_damped_v2"):
        original = build(mode)
        original.step(real)
        original.step(real.flip(0))
        saved = original.state_dict()
        restored = build(mode)
        restored.load_state_dict(saved)
        history = original._gradient_response
        other_history = restored._gradient_response
        original_count = None if history is None else history.packet["count"]
        restored_count = (None if other_history is None or other_history.packet is None
                          else other_history.packet["count"])
        original.step(real)
        restored.step(real)
        difference = max(float((p.detach() - q.detach()).abs().max())
                         for p, q in zip(original.G.parameters(), restored.G.parameters()))
        if mode == "none":
            assert difference == 0
        else:
            assert original_count == 2 and restored_count is None and difference > 0
        print(json.dumps({"check": "response_resume", "mode": mode,
                          "original_history_count": original_count,
                          "restored_history_count": restored_count,
                          "next_generator_max_abs_difference": difference}), flush=True)

    trainer = build()
    captured = {}

    def callback():
        captured["phase"] = trainer.policy._phase
        captured["checkpoint"] = trainer.state_dict()
        return real

    trainer.step(real, generator_real=callback)
    saved = captured["checkpoint"]
    critic_steps = saved["optimizers"][1]["regularizer"]["record"]["observed_steps"]
    assert captured["phase"] == "generator" and saved["completed_steps"] == 0 and critic_steps == 1
    print(json.dumps({"check": "mid_step_checkpoint", "accepted": True,
                      "phase": captured["phase"], "saved_completed_steps": saved["completed_steps"],
                      "saved_critic_observed_steps": critic_steps}), flush=True)
    try:
        build().load_state_dict(saved)
    except ValueError as error:
        assert "requires_grad" in str(error)
        print(json.dumps({"check": "mid_step_restore", "result": "ValueError", "reason": str(error)}), flush=True)
    else:
        raise AssertionError("audited mid-step checkpoint unexpectedly restored")


if __name__ == "__main__":
    main()
