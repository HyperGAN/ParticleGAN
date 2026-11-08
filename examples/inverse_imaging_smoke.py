"""Three-update public-API inverse-imaging integration, with actual training frames.

This checks finite execution, parameter movement and checkpoint replay. It is a
software integration fixture, not a six-gate imaging-quality benchmark. The older
seed-42 custom-Adam fixture in test_dataless_tomographic_inverse_e2e remains a
separate cohort. No image-fidelity or Forge qualification is claimed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from particlegan import GANTrainer, get_recipe
from particlegan.init import deterministic_orthogonal_
from particlegan.wavefront import (
    SimplexWavefrontGenerator,
    SobolevGradFilter,
    SplatCarrierManipulator,
)

STEPS = 3


class ObservableGenerator(nn.Module):
    """Expose left/right image means to a scalar GAN critic through a fixed operator."""

    def __init__(self):
        super().__init__()
        self.images = SimplexWavefrontGenerator(channels=1, hidden_dim=4, num_blocks=1)
        self.backward_filter = SobolevGradFilter(lam=1.0)

    def forward(self, codes):
        images = self.backward_filter(self.images(codes))
        return torch.stack((images[..., :4].mean((1, 2, 3)),
                            images[..., 4:].mean((1, 2, 3))), dim=1)


def _build():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        recipe = get_recipe("bcap", num_particles=8, z_dim=2, batch_size=8,
                            total_steps=STEPS, prior_reg=0., training_precision="float32")
        generator = ObservableGenerator()
        critic = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
        prior = recipe.make_prior().manipulate(SplatCarrierManipulator(
            num_modes=2, channels=1, height=8, width=8,
            num_base_splats=2, num_detail_splats=2))
        for seed, module in enumerate((generator, critic, prior)):
            deterministic_orthogonal_(module, seed=seed)
        return GANTrainer(recipe, generator, critic, prior=prior, seed=0)


def _batch(generator):
    index = torch.randint(2, (8,), generator=generator)
    centers = torch.tensor([[.5, -.5], [-.5, .5]])
    return centers[index] + .01 * torch.randn(8, 2, generator=generator)


def _same_state(left, right):
    if isinstance(left, torch.Tensor):
        try:
            torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
            return True
        except (AssertionError, TypeError):
            return False
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(_same_state(value, right[key]) for key, value in left.items()))
    if isinstance(left, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_same_state(a, b) for a, b in zip(left, right)))
    if isinstance(left, float) and math.isnan(left):
        return isinstance(right, float) and math.isnan(right)
    return left == right


def run(output: Path):
    from PIL import Image, ImageDraw

    output.mkdir(parents=True, exist_ok=True)
    trainer = _build()
    data_rng = torch.Generator().manual_seed(100)
    eval_rng = torch.Generator().manual_seed(101)
    eval_noise_rng = torch.Generator().manual_seed(102)
    initial = deepcopy(trainer.G.state_dict())
    frames, losses = [], []

    def observe(step):
        # Reusing fixed evaluation noise isolates visible training changes.
        eval_rng.manual_seed(101)
        eval_noise_rng.manual_seed(102)
        with torch.no_grad():
            codes, _ = trainer.prior.sample(4, generator=eval_rng, noise_generator=eval_noise_rng)
            images = trainer.G.images(codes).squeeze(1)
        canvas = Image.new("RGB", (576, 192), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f"Public GANTrainer integration: update {step}/{STEPS}", fill="black")
        draw.text((8, 30), "Target observable patterns | actual model images", fill="black")
        targets = torch.stack((torch.cat((torch.full((8, 4), .5), torch.full((8, 4), -.5)), 1),
                               torch.cat((torch.full((8, 4), -.5), torch.full((8, 4), .5)), 1)))
        for index, picture in enumerate(torch.cat((targets, images))):
            pixels = ((picture.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).numpy()
            tile = Image.fromarray(pixels).convert("RGB").resize((80, 80), Image.Resampling.NEAREST)
            canvas.paste(tile, (8 + index * 94, 58))
        draw.text((8, 154), "Gate: finite updates + changed parameters + exact replay", fill="black")
        draw.text((8, 173), "Integration only; image quality is not graded.", fill="black")
        frames.append(canvas)

    observe(0)
    checkpoint = None
    for step in range(STEPS):
        losses.append(trainer.step(_batch(data_rng)))
        if step == 1:
            checkpoint = deepcopy(trainer.state_dict()), data_rng.get_state().clone()
        observe(step + 1)
    movement = max(float((value - initial[name]).abs().max())
                   for name, value in trainer.G.state_dict().items() if value.is_floating_point())
    expected_state = deepcopy(trainer.state_dict())
    saved, data_state = checkpoint
    resumed = _build()
    resumed.load_state_dict(saved)
    resumed_data = torch.Generator().set_state(data_state)
    replayed = resumed.step(_batch(resumed_data))
    replay_exact = (_same_state(replayed, losses[-1])
                    and _same_state(expected_state, resumed.state_dict())
                    and torch.equal(data_rng.get_state(), resumed_data.get_state()))
    finite = all(torch.isfinite(torch.tensor([row["loss_g"], row["loss_d"]])).all().item() for row in losses)
    passed = finite and movement > 0 and replay_exact and trainer.completed_steps == STEPS
    gif = output / "public-api-integration.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=800, loop=0)
    result = {"scope": "public_api_inverse_imaging_integration_v1", "protocol_seed": 0,
              "initializer": "deterministic_orthogonal", "updates": STEPS,
              "metric_gate": {"all_losses_finite": finite, "maximum_parameter_change_gt_0": movement,
                              "checkpoint_replay_exact": replay_exact},
              "verdict": "PASS" if passed else "FAIL", "qualification_input": False,
              "image_quality_graded": False, "recipe": trainer.recipe.to_dict(),
              "data_rng_state_sha256": hashlib.sha256(data_rng.get_state().numpy().tobytes()).hexdigest(),
              "gif_sha256": hashlib.sha256(gif.read_bytes()).hexdigest()}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("scope", "verdict", "metric_gate")}), flush=True)
    if not passed:
        raise RuntimeError("public API inverse-imaging integration failed")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/inverse-imaging-integration"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.output)
