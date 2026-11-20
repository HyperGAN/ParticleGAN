"""Reproducible CIFAR-style benchmark runner and leaderboard formatter.

Adheres strictly to ``AGENTS.md`` comparison rules:
- Protocol seed 0 and ``particlegan.init.deterministic_orthogonal_``.
- Fixed architecture, dataset, critic/generator target batch sequence, initial
  prior distribution/capacity, update budget, and evaluation cadence across
  trainers.
- Isolated constructor, data, training-noise, prior-noise, and eval RNG streams.
- Executes exact resolved ParticleGAN ``Recipe`` via ``GANTrainer`` with zero
  task-host repairs, gradient edits, or silent substitutions.
- Tail-friendly flushed single-line stdout logs + JSONL trace + compact JSON
  receipts + single current markdown leaderboard per run directory.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

import particlegan as pg

from .data import (
    BatchSequenceStream,
    CIFAR10Dataset,
    compute_class_centroids,
    load_cifar10_dataset,
    make_synthetic_cifar_dataset,
    sample_indices_for_eval,
    select_eval_reference_subset,
)
from .metrics import (
    compute_class_coverage,
    compute_critic_diagnostics,
    compute_fid,
    compute_grad_norm,
    compute_kid,
    compute_pixel_diversity_ratio,
    feature_moments,
)
from .models import (
    DEFAULT_RESNET18_WEIGHTS,
    DEFAULT_RESNET34_WEIGHTS,
    ResNet18FeatureCritic32,
    ResNet34FeatureEvaluator,
    TinyTransformerGenerator32,
    count_matrix_conv_params,
    count_trainable_params,
    parameter_state_digest,
)

# Exact fixed MoG sigma from HyperGAN cifar10-tiny-transformer (0.025 * median NN in z=64).
DEFAULT_MOG_SIGMA = 0.212616428732872


@dataclass(frozen=True)
class BenchmarkConfig:
    recipe: str = "unified_v11"
    num_particles: int = 200
    z_dim: int = 64
    mog_sigma: float = DEFAULT_MOG_SIGMA
    standardize_prior: bool = True
    batch_size: int = 128
    steps: int = 800
    eval_every: int = 200
    eval_samples: int = 2048
    eval_ref_samples: int = 5000
    eval_critic_samples: int = 256
    seed: int = 0
    dataset: str = "cifar10"
    data_root: Optional[str] = None
    horizontal_flip: bool = True
    backbone_mode: str = "pretrained"
    resnet18_weights: str = str(DEFAULT_RESNET18_WEIGHTS)
    resnet34_weights: str = str(DEFAULT_RESNET34_WEIGHTS)
    device: str = "cuda:0"
    deterministic_algorithms: bool = False
    recipe_overrides: Dict[str, Any] = field(default_factory=dict)
    run_tag: Optional[str] = None


def _git_info() -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    info: Dict[str, Any] = {"root": str(root)}
    for key, cmd in (
        ("commit", ["git", "rev-parse", "HEAD"]),
        ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        ("dirty", ["git", "status", "--porcelain"]),
    ):
        try:
            res = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                val = res.stdout.strip()
                info[key] = bool(val) if key == "dirty" else val
        except OSError:
            pass
    return info


def _resolve_device(device_str: str) -> torch.device:
    dev = torch.device(device_str)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device_str} but CUDA is unavailable")
        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or platform.machine() or "cpu"


def _capture_backend_state(device: torch.device) -> dict:
    state: dict = {
        "cpu_rng": torch.get_rng_state(),
        "num_threads": torch.get_num_threads(),
        "det_alg": torch.are_deterministic_algorithms_enabled(),
        "cudnn_det": torch.backends.cudnn.deterministic,
        "cudnn_bench": torch.backends.cudnn.benchmark,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    if device.type == "cuda":
        state["cuda_rng"] = torch.cuda.get_rng_state(device)
    return state


def _restore_backend_state(state: dict, device: torch.device) -> None:
    torch.set_rng_state(state["cpu_rng"])
    torch.set_num_threads(state["num_threads"])
    torch.use_deterministic_algorithms(state["det_alg"])
    torch.backends.cudnn.deterministic = state["cudnn_det"]
    torch.backends.cudnn.benchmark = state["cudnn_bench"]
    torch.backends.cudnn.allow_tf32 = state["cudnn_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = state["matmul_tf32"]
    if device.type == "cuda" and "cuda_rng" in state:
        torch.cuda.set_rng_state(state["cuda_rng"], device)


def _format_progress_line(run_name: str, step: int, total_steps: int, row: Mapping[str, Any]) -> str:
    fid = row.get("feature_fid", float("nan"))
    kid = row.get("feature_kid_x1e3", float("nan"))
    cov = int(round(row.get("class_coverage", 0.0)))
    acc = row.get("critic_accuracy", float("nan"))
    margin = row.get("critic_margin", float("nan"))
    r_m = row.get("critic_real_mean", float("nan"))
    f_m = row.get("critic_fake_mean", float("nan"))
    gn = row.get("grad_norm_g", float("nan"))
    dn = row.get("grad_norm_d", float("nan"))
    pn = row.get("grad_norm_prior", float("nan"))
    ips = row.get("steps_per_sec", float("nan"))
    vram = row.get("peak_vram_mib", 0.0)
    return (
        f"[{run_name}] step={step:04d}/{total_steps:04d} "
        f"fid={fid:6.2f} kid={kid:6.2f} cov={cov:2d}/10 "
        f"acc={acc:.3f} margin={margin:+.3f} (r={r_m:+.3f},f={f_m:+.3f}) "
        f"gn={gn:.3f} dn={dn:.3f} pn={pn:.4f} "
        f"ips={ips:5.2f} vram={vram:6.0f}MiB"
    )


def _load_dataset_for_config(cfg: BenchmarkConfig) -> CIFAR10Dataset:
    if cfg.dataset == "cifar10":
        return load_cifar10_dataset(Path(cfg.data_root) if cfg.data_root else None, split="train")
    if cfg.dataset == "synthetic":
        return make_synthetic_cifar_dataset(
            num_samples=max(2000, cfg.eval_ref_samples),
            seed=cfg.seed,
            num_classes=10,
        )
    raise ValueError(f"Unknown dataset {cfg.dataset!r}; expected 'cifar10' or 'synthetic'")


def run_benchmark(
    cfg: BenchmarkConfig,
    output_dir: Path,
    *,
    dataset: Optional[CIFAR10Dataset] = None,
    evaluator: Optional[ResNet34FeatureEvaluator] = None,
    ref_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute one matched benchmark run and write tail-friendly logs + receipt."""
    device = _resolve_device(cfg.device)
    out_root = Path(output_dir)
    run_name = cfg.run_tag or f"{cfg.recipe}_p{cfg.num_particles}_b{cfg.batch_size}_s{cfg.steps}"
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    log_path = run_dir / "train.log"
    receipt_path = run_dir / "receipt.json"

    saved_state = _capture_backend_state(device)
    try:
        # Fixed protocol execution hygiene (restored in finally).
        torch.manual_seed(cfg.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(cfg.seed)
            torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cudnn.deterministic = bool(cfg.deterministic_algorithms)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        if cfg.deterministic_algorithms:
            torch.use_deterministic_algorithms(True)

        ds = _load_dataset_for_config(cfg) if dataset is None else dataset

        # Prepare independent ResNet-34 evaluator and reference feature moments once.
        own_eval = evaluator is None
        if own_eval:
            evaluator = ResNet34FeatureEvaluator(
                Path(cfg.resnet34_weights),
                backbone_mode=cfg.backbone_mode,
                backbone_seed=cfg.seed + 2002,
            ).to(device=device)
        evaluator.eval()

        if ref_cache is None:
            ref_imgs, ref_labels = select_eval_reference_subset(ds, cfg.eval_ref_samples, seed=cfg.seed)
            ref_feats = evaluator.extract(ref_imgs, batch_size=256)
            ref_mu, ref_cov = feature_moments(ref_feats)
            ref_centroids = compute_class_centroids(ref_feats, ref_labels, num_classes=ds.num_classes)
            eval_critic_real = ref_imgs[: cfg.eval_critic_samples].clone()
            ref_cache = {
                "ref_imgs": ref_imgs,
                "ref_feats": ref_feats,
                "ref_mu": ref_mu,
                "ref_cov": ref_cov,
                "ref_centroids": ref_centroids,
                "eval_critic_real": eval_critic_real,
            }

        # Construct fixed task architecture & deterministic initialization.
        G = TinyTransformerGenerator32(z_dim=cfg.z_dim).to(device=device)
        D = ResNet18FeatureCritic32(
            Path(cfg.resnet18_weights),
            backbone_mode=cfg.backbone_mode,
            backbone_seed=cfg.seed + 1001,
        ).to(device=device)
        prior = pg.MoGParticlePrior(
            cfg.num_particles,
            cfg.z_dim,
            sigma=cfg.mog_sigma,
            standardize=cfg.standardize_prior,
            device=device,
        )

        # Deterministic public initializer (seed=0 for G & prior, seed=1 for D).
        pg.init.deterministic_orthogonal_(G, seed=cfg.seed, strict=True)
        pg.init.deterministic_orthogonal_(D, seed=cfg.seed + 1, strict=True)
        pg.init.deterministic_orthogonal_(prior, seed=cfg.seed, strict=True)

        init_digests = {
            "initial_g_sha256": parameter_state_digest(G),
            "initial_d_sha256": parameter_state_digest(D),
            "initial_prior_sha256": parameter_state_digest(prior),
            "g_trainable_params": count_trainable_params(G),
            "g_matrix_conv_params": count_matrix_conv_params(G),
            "d_trainable_params": count_trainable_params(D),
            "d_matrix_conv_params": count_matrix_conv_params(D),
            "d_backbone_sha256": D.backbone_sha256,
            "eval_backbone_sha256": evaluator.backbone_sha256,
        }

        # Resolve exact public ParticleGAN Recipe (no host repairs / overrides unless explicitly declared).
        recipe_kwargs: Dict[str, Any] = {
            "prior_kind": "mog",
            "z_dim": cfg.z_dim,
            "num_particles": cfg.num_particles,
            "sigma_rel": 0.0,
            "standardize": cfg.standardize_prior,
            "batch_size": cfg.batch_size,
            "total_steps": cfg.steps,
        }
        recipe_kwargs.update(cfg.recipe_overrides)
        recipe = pg.get_recipe(cfg.recipe, **recipe_kwargs)

        # Isolated named RNG streams on execution device.
        def _dev_gen(offset: int) -> torch.Generator:
            return torch.Generator(device=device).manual_seed(cfg.seed + offset)

        trainer = pg.GANTrainer(
            recipe,
            G,
            D,
            prior=prior,
            seed=cfg.seed,
            latent_generator=_dev_gen(11),
            penalty_generator=_dev_gen(12),
            noise_generator=_dev_gen(13),
            input_noise_generator=_dev_gen(14),
            prior_noise_generator=_dev_gen(15),
            eval_generator=_dev_gen(16),
            max_steps=cfg.steps,
        )

        # Isolated CPU data streams guaranteeing identical (real_d, real_g) sequence across recipes.
        data_stream = BatchSequenceStream(
            ds,
            cfg.batch_size,
            seed=cfg.seed,
            horizontal_flip=cfg.horizontal_flip,
        )

        def _evaluate(step: int, last_step_info: Dict[str, float], train_seconds: float) -> Dict[str, Any]:
            # Fixed per-step eval RNG so sampling is deterministic and isolated from training.
            eval_stream = torch.Generator(device=device).manual_seed(cfg.seed + 50000 + step)
            fake_chunks = []
            for b in sample_indices_for_eval(cfg.eval_samples, min(256, cfg.eval_samples)):
                samples = trainer.sample(b, generator=eval_stream, output_noise=False)
                fake_chunks.append(samples.detach().float().cpu())
            fake_all = torch.cat(fake_chunks, dim=0)

            fake_feats = evaluator.extract(fake_all, batch_size=256)
            fake_mu, fake_cov = feature_moments(fake_feats)
            fid = compute_fid(ref_cache["ref_mu"], ref_cache["ref_cov"], fake_mu, fake_cov)
            kid_mean, kid_std = compute_kid(
                ref_cache["ref_feats"],
                fake_feats,
                subset_size=min(500, cfg.eval_samples, cfg.eval_ref_samples),
                num_subsets=20,
                seed=cfg.seed,
            )
            cov_stats = compute_class_coverage(fake_feats, ref_cache["ref_centroids"])
            div_ratio = compute_pixel_diversity_ratio(
                ref_cache["eval_critic_real"], fake_all[: cfg.eval_critic_samples]
            )
            critic_diag = compute_critic_diagnostics(
                D,
                ref_cache["eval_critic_real"],
                fake_all[: cfg.eval_critic_samples],
                batch_size=min(128, cfg.eval_critic_samples),
            )
            peak_vram_mib = (
                float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
                if device.type == "cuda"
                else 0.0
            )
            ips = float(step) / train_seconds if (step > 0 and train_seconds > 0.0) else 0.0
            row: Dict[str, Any] = {
                "step": step,
                "feature_fid": fid,
                "feature_kid_x1e3": kid_mean * 1000.0,
                "feature_kid_std_x1e3": kid_std * 1000.0,
                **cov_stats,
                "diversity_ratio": div_ratio,
                **critic_diag,
                **last_step_info,
                "elapsed_train_sec": train_seconds,
                "steps_per_sec": ips,
                "peak_vram_mib": peak_vram_mib,
            }
            return row

        history: List[Dict[str, Any]] = []
        status = "COMPLETE"
        error_msg: Optional[str] = None
        train_seconds = 0.0
        last_step_info: Dict[str, float] = {
            "loss_d": float("nan"),
            "loss_g": float("nan"),
            "loss_gan": float("nan"),
            "penalty": float("nan"),
            "prior_reg": float("nan"),
            "grad_norm_g": 0.0,
            "grad_norm_d": 0.0,
            "grad_norm_prior": 0.0,
            "prior_rms": float(prior.z.detach().float().square().mean().sqrt().item()),
        }

        with open(metrics_path, "w", encoding="utf-8") as mf, open(log_path, "w", encoding="utf-8") as lf:
            # Step-0 evaluation (verifies matched initialization across recipes).
            row0 = _evaluate(0, last_step_info, 0.0)
            history.append(row0)
            mf.write(json.dumps(row0) + "\n")
            mf.flush()
            line0 = _format_progress_line(run_name, 0, cfg.steps, row0)
            print(line0, flush=True)
            lf.write(line0 + "\n")
            lf.flush()

            for step in range(1, cfg.steps + 1):
                real_d, real_g = data_stream.next_pair(device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                out = trainer.step(real_d, generator_real=real_g)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                train_seconds += time.perf_counter() - t0

                loss_d = float(out["loss_d"].item())
                loss_g = float(out["loss_g"].item())
                loss_gan = float(out["loss_gan"].item())
                pen = float(out["penalty"].item())
                preg = float(out["prior_regularization"].item())

                if not (math.isfinite(loss_d) and math.isfinite(loss_g)):
                    status = "NONFINITE"
                    error_msg = f"Nonfinite loss at step {step}: loss_d={loss_d}, loss_g={loss_g}"
                    lf.write(f"ERROR: {error_msg}\n")
                    lf.flush()
                    print(f"[{run_name}] ERROR: {error_msg}", flush=True)
                    break

                last_step_info = {
                    "loss_d": loss_d,
                    "loss_g": loss_g,
                    "loss_gan": loss_gan,
                    "penalty": pen,
                    "prior_reg": preg,
                    "grad_norm_g": compute_grad_norm(G.parameters()),
                    "grad_norm_d": compute_grad_norm(D.parameters()),
                    "grad_norm_prior": compute_grad_norm([prior.z]),
                    "prior_rms": float(prior.z.detach().float().square().mean().sqrt().item()),
                }

                is_eval = (step % cfg.eval_every == 0) or (step == cfg.steps)
                if is_eval:
                    row = _evaluate(step, last_step_info, train_seconds)
                    history.append(row)
                    mf.write(json.dumps(row) + "\n")
                    mf.flush()
                    line = _format_progress_line(run_name, step, cfg.steps, row)
                    print(line, flush=True)
                    lf.write(line + "\n")
                    lf.flush()

        # Summarize best & final metrics (excluding step 0 for best trained FID/KID).
        trained_rows = [r for r in history if r["step"] > 0]
        final_row = history[-1]
        best_fid_row = min(trained_rows, key=lambda r: r["feature_fid"]) if trained_rows else final_row
        best_kid_row = min(trained_rows, key=lambda r: r["feature_kid_x1e3"]) if trained_rows else final_row

        receipt: Dict[str, Any] = {
            "schema_version": 1,
            "run_name": run_name,
            "status": status,
            "error": error_msg,
            "config": asdict(cfg),
            "resolved_recipe": recipe.to_dict(),
            "dataset": {
                "source_name": ds.source_name,
                "source_path": ds.source_path,
                "size": ds.size,
                "num_classes": ds.num_classes,
            },
            "hardware": {
                "device": str(device),
                "device_name": _device_name(device),
                "torch_version": torch.__version__,
                "python_version": platform.python_version(),
            },
            "git": _git_info(),
            "provenance": {
                **init_digests,
                "batch_sequence_digest": data_stream.digest,
                "steps_completed": int(final_row["step"]),
                "sampling_law": "clean_live_mog",
                "eval_output_noise": False,
            },
            "summary": {
                "final_step": int(final_row["step"]),
                "final_fid": float(final_row["feature_fid"]),
                "best_fid": float(best_fid_row["feature_fid"]),
                "best_fid_step": int(best_fid_row["step"]),
                "final_kid_x1e3": float(final_row["feature_kid_x1e3"]),
                "best_kid_x1e3": float(best_kid_row["feature_kid_x1e3"]),
                "final_class_coverage": int(round(final_row["class_coverage"])),
                "final_class_mass_tv": float(final_row["class_mass_tv"]),
                "final_diversity_ratio": float(final_row["diversity_ratio"]),
                "final_critic_accuracy": float(final_row["critic_accuracy"]),
                "final_critic_margin": float(final_row["critic_margin"]),
                "final_critic_real_mean": float(final_row["critic_real_mean"]),
                "final_critic_fake_mean": float(final_row["critic_fake_mean"]),
                "final_grad_norm_g": float(final_row["grad_norm_g"]),
                "final_grad_norm_d": float(final_row["grad_norm_d"]),
                "final_grad_norm_prior": float(final_row["grad_norm_prior"]),
                "steps_per_sec": float(final_row["steps_per_sec"]),
                "elapsed_train_sec": float(final_row["elapsed_train_sec"]),
                "peak_vram_mib": float(final_row["peak_vram_mib"]),
            },
            "history": history,
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return receipt
    finally:
        _restore_backend_state(saved_state, device)


def format_leaderboard_markdown(receipts: Sequence[Mapping[str, Any]]) -> str:
    """Render a compact, decision-ready markdown leaderboard table from receipts."""
    ordered = sorted(
        receipts,
        key=lambda r: (
            0 if r.get("status") == "COMPLETE" else 1,
            float(r.get("summary", {}).get("final_fid", float("inf"))),
        ),
    )
    lines = [
        "| Run | Recipe | MoG N | Status | Final FID ↓ | Best FID ↓ (step) | KID×1e3 ↓ | Cov /10 | TV ↓ | D acc | Margin | G grad | D grad | it/s ↑ | Peak MiB |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in ordered:
        cfg = r.get("config", {})
        s = r.get("summary", {})
        lines.append(
            f"| `{r.get('run_name', '')}` | `{cfg.get('recipe', '')}` | {cfg.get('num_particles', 0)} "
            f"| {r.get('status', 'UNKNOWN')} "
            f"| {s.get('final_fid', float('nan')):.2f} "
            f"| {s.get('best_fid', float('nan')):.2f} ({s.get('best_fid_step', 0)}) "
            f"| {s.get('final_kid_x1e3', float('nan')):.2f} "
            f"| {s.get('final_class_coverage', 0)}/10 "
            f"| {s.get('final_class_mass_tv', float('nan')):.3f} "
            f"| {s.get('final_critic_accuracy', float('nan')):.3f} "
            f"| {s.get('final_critic_margin', float('nan')):+.3f} "
            f"| {s.get('final_grad_norm_g', float('nan')):.3f} "
            f"| {s.get('final_grad_norm_d', float('nan')):.3f} "
            f"| {s.get('steps_per_sec', float('nan')):.2f} "
            f"| {s.get('peak_vram_mib', 0.0):.0f} |"
        )
    return "\n".join(lines) + "\n"


def summarize_run_directory(output_dir: Path) -> Tuple[List[Dict[str, Any]], str]:
    """Collect all ``receipt.json`` files under ``output_dir`` and write one ``leaderboard.md``."""
    root = Path(output_dir)
    receipts = []
    for path in sorted(root.glob("*/receipt.json")):
        receipts.append(json.loads(path.read_text(encoding="utf-8")))
    md = format_leaderboard_markdown(receipts)
    if receipts:
        (root / "leaderboard.md").write_text(md, encoding="utf-8")
    return receipts, md
