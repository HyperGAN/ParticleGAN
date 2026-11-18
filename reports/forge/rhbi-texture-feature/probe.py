"""Read-only output-space probe of the published RHBI-v12 texture endpoint."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmarks.toy_audit.api_conv_texture32 import build_models_for_spec, sample_conv_texture32
from experiments.forge.contracts import file_hash
from experiments.forge.conv_texture32 import conv_texture32_context
from experiments.forge.planning import load_idea


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    digest = file_hash(args.checkpoint)
    if digest != "b5e96c96fa804fad5f31f6af84e5f735dcb1df537ae9bb746dbd056084d5409e":
        raise ValueError("Expected the original 288-update RHBI-v12 provenance checkpoint")
    root = Path(__file__).resolve().parents[3]
    task = json.loads((root / "configs/forge/tasks/img_conv_texture32.json").read_text())
    spec = task["execution"]["host_definition"]
    device = "cuda:1"  # Original named-stream device.
    ctx = conv_texture32_context(
        {"candidate": load_idea(root, "rhbi-v12"), "protocol": {"seed": 0}}, task, device,
    )
    def construct(index, role):
        return ctx.construct(lambda: build_models_for_spec(spec, z_dim=ctx.recipe.z_dim)[index],
                             component=role).to(device)

    trainer = ctx.build_trainer(construct(0, "generator"), construct(1, "discriminator"), max_steps=288)
    ctx.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    trainer.D.eval()
    rng = torch.Generator().manual_seed(0)
    ref = sample_conv_texture32(64, rng).to(device)
    real = sample_conv_texture32(64, rng).to(device)
    flat = real.mean(dim=(-2, -1), keepdim=True).expand_as(real)
    blurred = F.avg_pool2d(F.pad(real, (3, 3, 3, 3), mode="circular"), 7, stride=1)
    rec = trainer.opt_d.record
    rec._in_protected_probe = True
    features = []
    head = trainer.D.head[-1]

    def capture(_module, args):
        features.append(args[0])

    results = {}
    for name, sample in (("real", real), ("blurred", blurred), ("flat_mean", flat)):
        x = sample.detach().contiguous().requires_grad_()
        features.clear()
        with head.register_forward_pre_hook(capture):
            scores = trainer.D(x)
            target_scores = trainer.D(ref).detach()
        fx, fy = features
        # Same head coordinates and detached real RMS radius as the proposed arm.
        radius = (fy - fy.mean(0)).square().sum(1).mean().sqrt().detach().clamp_min(1e-6)
        feature_nn = torch.cdist(fx / radius, fy / radius).min(1).values
        loss, metrics = trainer.loss.rhbi(
            x, ref, scores, target_scores, record=rec, allow_ell_update=False,
        )
        critic_grad, = torch.autograd.grad(scores.mean(), x, retain_graph=True)
        loss_grad, = torch.autograd.grad(loss, x)
        restore = real - x.detach()
        distances = torch.cdist(x.detach().flatten(1), ref.flatten(1))
        results[name] = dict(
            **metrics, score=scores.mean().item(), raw_nn=distances.min(1).values.median().item(),
            generator_loss=loss.item(), feature_nn=feature_nn.median().item(),
            critic_ascent_toward_original=(critic_grad * restore).sum().item(),
            generator_descent_toward_original=(-loss_grad * restore).sum().item(),
        )
    print(json.dumps({"checkpoint_sha256": digest, "optimizer_updates_added": 0,
                      "probe": results}, indent=2))


if __name__ == "__main__":
    main()
