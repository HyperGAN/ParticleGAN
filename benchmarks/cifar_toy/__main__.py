"""CLI entrypoint: ``python -m benchmarks.cifar_toy``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .models import DEFAULT_RESNET18_WEIGHTS, DEFAULT_RESNET34_WEIGHTS
from .runner import (
    DEFAULT_MOG_SIGMA,
    BenchmarkConfig,
    format_leaderboard_markdown,
    run_benchmark,
    summarize_run_directory,
)


def _parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"Expected KEY=VALUE for --recipe-override, got {text!r}")
    key, raw = text.split("=", 1)
    key = key.strip()
    raw = raw.strip()
    try:
        val = json.loads(raw)
    except json.JSONDecodeError:
        val = raw
    return key, val


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m benchmarks.cifar_toy",
        description="Fast reproducible CIFAR-style 32x32 Transformer-G / ResNet18-feature-D ParticleGAN benchmark.",
    )
    p.add_argument(
        "--recipes",
        nargs="+",
        default=["bispf_optimized", "empf", "unified_v11"],
        help="ParticleGAN recipe presets to compare (default: bispf_optimized empf unified_v11).",
    )
    p.add_argument(
        "--num-particles",
        nargs="+",
        type=int,
        default=[200],
        help="One or more MoG prior particle counts (e.g. --num-particles 200 16384).",
    )
    p.add_argument("--z-dim", type=int, default=64)
    p.add_argument("--mog-sigma", type=float, default=DEFAULT_MOG_SIGMA)
    p.add_argument("--no-standardize-prior", action="store_true")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-samples", type=int, default=2048)
    p.add_argument("--eval-ref-samples", type=int, default=5000)
    p.add_argument("--eval-critic-samples", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", choices=("cifar10", "synthetic"), default="cifar10")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--no-horizontal-flip", action="store_true")
    p.add_argument("--backbone-mode", choices=("pretrained", "random"), default="pretrained")
    p.add_argument("--resnet18-weights", type=str, default=str(DEFAULT_RESNET18_WEIGHTS))
    p.add_argument("--resnet34-weights", type=str, default=str(DEFAULT_RESNET34_WEIGHTS))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--deterministic-algorithms", action="store_true")
    p.add_argument(
        "--recipe-override",
        action="append",
        default=[],
        help="Explicit recipe field override KEY=VALUE (JSON-parsed; repeatable).",
    )
    p.add_argument("--run-tag", type=str, default=None, help="Explicit run directory name (single run only).")
    p.add_argument(
        "--output-dir",
        type=str,
        default="/nvme/dev/hypergan/ParticleGAN/runs/cifar_toy_bench",
        help="Directory for bulk logs, JSONL traces, receipts, and leaderboard.md.",
    )
    p.add_argument(
        "--summarize-only",
        action="store_true",
        help="Do not train; rebuild leaderboard.md from existing receipts in --output-dir.",
    )
    return p


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out_dir = Path(args.output_dir)

    if args.summarize_only:
        receipts, md = summarize_run_directory(out_dir)
        if not receipts:
            print(f"No receipts found under {out_dir}")
            return 1
        print(md, end="")
        return 0

    overrides: Dict[str, Any] = {}
    for item in args.recipe_override:
        k, v = _parse_override(item)
        overrides[k] = v

    combos = [(r, n) for r in args.recipes for n in args.num_particles]
    if args.run_tag is not None and len(combos) != 1:
        parser.error("--run-tag can only be used with a single (recipe, num_particles) combination")

    receipts = []
    for recipe_name, num_p in combos:
        cfg = BenchmarkConfig(
            recipe=recipe_name,
            num_particles=num_p,
            z_dim=args.z_dim,
            mog_sigma=args.mog_sigma,
            standardize_prior=not args.no_standardize_prior,
            batch_size=args.batch_size,
            steps=args.steps,
            eval_every=args.eval_every,
            eval_samples=args.eval_samples,
            eval_ref_samples=args.eval_ref_samples,
            eval_critic_samples=args.eval_critic_samples,
            seed=args.seed,
            dataset=args.dataset,
            data_root=args.data_root,
            horizontal_flip=not args.no_horizontal_flip,
            backbone_mode=args.backbone_mode,
            resnet18_weights=args.resnet18_weights,
            resnet34_weights=args.resnet34_weights,
            device=args.device,
            deterministic_algorithms=args.deterministic_algorithms,
            recipe_overrides=overrides,
            run_tag=args.run_tag,
        )
        receipts.append(run_benchmark(cfg, out_dir))

    _, md = summarize_run_directory(out_dir)
    print("\n### Current Leaderboard\n", flush=True)
    print(format_leaderboard_markdown(receipts), end="", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
