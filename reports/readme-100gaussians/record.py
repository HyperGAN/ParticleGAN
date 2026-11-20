"""Record actual training through the current public default; keep raw observations local."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10; included in the experiments extra.
    import tomli as tomllib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from particlegan import GANTrainer, get_recipe, init
from lib.denoising_toy import GaussianGrid, grid_metrics
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator, sample_100gaussians

SEED, EVAL_SAMPLES, PLOT_SAMPLES = 0, 20_000, 4096
STREAM_SEEDS = dict(constructor=11, data=12, latent_generator=13,
                    penalty_generator=14, noise_generator=15,
                    input_noise_generator=16, eval_generator=17,
                    model_generator=18, reference=19)


def frame_steps(total, frames, power):
    """Strictly increasing observed updates, dense near the start, including the endpoint."""
    previous, steps = 0, []
    for i in range(1, frames + 1):
        previous = min(total, max(previous + 1, round(total * (i / frames) ** power)))
        steps.append(previous)
    return sorted(set(steps))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module_digest(module):
    result = hashlib.sha256()
    for name, value in module.state_dict().items():
        result.update(name.encode())
        if isinstance(value, torch.Tensor):
            result.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            result.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    return result.hexdigest()


@torch.no_grad()
def score(samples, centers):
    """20k draws; a covered mode has >=10 samples within Euclidean radius 3 sigma."""
    if not bool(torch.isfinite(samples).all()):
        raise ValueError("nonfinite model outputs")
    distance, nearest = torch.cdist(samples, centers).min(1)
    inside = distance <= .09
    covered = torch.bincount(nearest[inside], minlength=100)
    counts = torch.bincount(nearest, minlength=100).float()
    modes, hq = int((covered >= 10).sum()), float(inside.float().mean())
    return dict(modes=modes, hq=hq, mode_tv=float((counts / len(samples) - .01).abs().sum() / 2),
                passed=modes == 100 and hq >= .95)


def make_trainer(recipe, device, *, max_steps=None, prior_sigma=None):
    devices = [device.index] if device.type == "cuda" else []
    prior_kwargs = {"sigma": prior_sigma} if recipe.prior_kind == "mog" else {}
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(STREAM_SEEDS["constructor"])
        generator = init.deterministic_orthogonal_(SimpleMLPGenerator(z_dim=recipe.z_dim), seed=0)
        critic = init.deterministic_orthogonal_(SimpleMLPDiscriminator(fourier=2), seed=1)
        prior = init.deterministic_orthogonal_(recipe.make_prior(**prior_kwargs), seed=2)
        constructor_rng = torch.get_rng_state().clone()
    streams = {name: torch.Generator(device=device).manual_seed(seed)
               for name, seed in STREAM_SEEDS.items() if name.endswith("_generator")}
    trainer = GANTrainer(recipe, generator.to(device), critic.to(device), prior=prior.to(device),
                         seed=SEED, max_steps=max_steps, **streams)
    trainer.readme_constructor_rng = constructor_rng
    return trainer


def _resolve_task_recipe(default, args, parser):
    """Apply only explicit task-owned prior/resource declarations; never touch technique fields."""
    prior_kind = args.prior_kind if args.prior_kind is not None else default.prior_kind
    standardize = args.standardize if args.standardize is not None else default.standardize
    num_particles = args.num_particles if args.num_particles is not None else default.num_particles
    batch_size = args.batch_size if args.batch_size is not None else default.batch_size
    if num_particles <= 0 or batch_size <= 0:
        parser.error("--num-particles and --batch-size must be positive integers")
    if prior_kind == "mog":
        prior_sigma = 0.025 if args.prior_sigma is None else args.prior_sigma
        if not np.isfinite(prior_sigma) or prior_sigma < 0.0:
            parser.error("--prior-sigma must be finite and nonnegative for prior_kind='mog'")
        recipe = default.replace(prior_kind="mog", sigma_rel=0.0, standardize=standardize,
                                 num_particles=num_particles, batch_size=batch_size)
        prior_spec = dict(kind="mog", class_name="MoGParticlePrior", num_particles=num_particles,
                          z_dim=recipe.z_dim, sigma=float(prior_sigma), standardize=bool(standardize),
                          learnable=recipe.prior_update == "learned")
    else:
        if args.prior_sigma is not None and args.prior_sigma != 0.0:
            parser.error("--prior-sigma requires --prior-kind mog")
        prior_sigma = 0.0
        recipe = default.replace(prior_kind="particles", standardize=standardize,
                                 num_particles=num_particles, batch_size=batch_size)
        prior_spec = dict(kind="particle_cloud", class_name="ParticlePrior", num_particles=num_particles,
                          z_dim=recipe.z_dim, sigma=0.0, standardize=bool(standardize),
                          learnable=recipe.prior_update == "learned")
    return recipe, prior_sigma, prior_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20_000,
                        help="External demo budget (default: 20000); recipe and schedules stay unchanged.")
    parser.add_argument("--resume-run", type=Path, help="Continue a saved recording, retaining all scored observations.")
    parser.add_argument("--prior-kind", choices=("particles", "mog"), default=None,
                        help="Explicit task-owned prior family (default: recipe.prior_kind).")
    parser.add_argument("--prior-sigma", type=float, default=None,
                        help="Intrinsic isotropic Gaussian sigma for --prior-kind mog (default: 0.025).")
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=None,
                        help="Explicit task-owned prior standardization flag (default: recipe.standardize).")
    parser.add_argument("--num-particles", type=int, default=None,
                        help="Explicit task-owned prior component/row count (default: recipe.num_particles).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Explicit task-owned batch size (default: recipe.batch_size).")
    parser.add_argument("--frames", type=int, default=250)
    parser.add_argument("--frame-power", type=float, default=2.)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--hold-seconds", type=float, default=2.5)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.device_count() > 1 else
                        ("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/100gaussians")
    parser.add_argument("--record-only", action="store_true")
    args = parser.parse_args()
    default = get_recipe()  # Follow the public default; never select a different favorite for the demo.
    recipe, prior_sigma, prior_spec = _resolve_task_recipe(default, args, parser)
    total = args.steps
    if total is None or total <= 0 or args.frames <= 0 or not np.isfinite(args.frame_power) or args.frame_power <= 0:
        parser.error("a positive budget, --frames and finite --frame-power are required")
    if args.fps <= 0 or 100 % args.fps or not np.isfinite(args.hold_seconds) or args.hold_seconds < 0:
        parser.error("--fps must divide 100 and --hold-seconds must be finite and nonnegative")
    if recipe.model != "gan" or recipe.conditioning != "scalar" or recipe.encoder_mode != "none":
        parser.error("the current default requires a different host; update this example explicitly")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(SEED)
    parent, prior_data, saved = None, None, None
    if args.resume_run is not None:
        parent_path = args.resume_run.resolve()
        parent = json.loads((parent_path / "summary.json").read_text())
        if (digest(parent_path / "final.pt") != parent["final_checkpoint_sha256"]
                or digest(parent_path / "observations.npz") != parent["observations_sha256"]):
            raise ValueError("resume artifacts do not match the parent receipt")
        if parent["recipe"] != json.loads(json.dumps(recipe.to_dict())):
            raise ValueError("resume recipe differs from the current public default")
        if "prior" in parent and parent["prior"] != prior_spec:
            raise ValueError("resume prior cohort differs from the requested prior")
        if (parent["seed"] != SEED or parent["stream_seeds"] != STREAM_SEEDS
                or parent["completed_steps"] >= total or parent["device"] != str(device)):
            raise ValueError("resume protocol/device differs or the new budget does not extend it")
        for name, expected in parent["source_sha256"].items():
            if name.startswith(("particlegan/", "lib/")) and digest(ROOT / name) != expected:
                raise ValueError(f"resume numerical source changed: {name}")
        saved = torch.load(parent_path / "final.pt", weights_only=False)
        with np.load(parent_path / "observations.npz", allow_pickle=False) as archive:
            prior_data = {name: archive[name] for name in archive.files}
        old_cap = saved["trainer"].get("max_steps", recipe.total_steps)
        trainer = make_trainer(recipe, device, max_steps=old_cap, prior_sigma=prior_sigma)
    else:
        trainer = make_trainer(recipe, device, max_steps=total, prior_sigma=prior_sigma)
    initial_hashes = {name: module_digest(module) for name, module in
                      [("G", trainer.G), ("D", trainer.D), ("prior", trainer.prior)]}
    data = torch.Generator(device=device).manual_seed(STREAM_SEEDS["data"])
    evaluation_initial = trainer.eval_generator.get_state().clone()
    reference = torch.Generator(device=device).manual_seed(STREAM_SEEDS["reference"])
    if saved is not None:
        if initial_hashes != parent["initial_sha256"]:
            raise ValueError("resume public initializer no longer matches the parent")
        trainer.load_state_dict(saved["trainer"])
        trainer.extend_execution(total)
        trainer.readme_constructor_rng = saved["constructor_rng"]
        data.set_state(saved["data_rng"])
        reference.set_state(saved["reference_rng"])
        evaluation_initial = saved["evaluation_initial"]
        target = torch.as_tensor(prior_data["target"], device=device)
    else:
        target = sample_100gaussians(EVAL_SAMPLES, device, generator=reference)
    start_step = trainer.completed_steps
    centers = torch.cartesian_prod(torch.arange(10, device=device), torch.arange(10, device=device)).float() - 4.5
    oracle, collapsed = score(target, centers), score(torch.zeros_like(target), centers)
    if not oracle["passed"] or collapsed["passed"]:
        raise ValueError("coverage scorer failed oracle/collapse controls")
    media = [0, *frame_steps(total, args.frames, args.frame_power)]
    checks = min(24, total - start_step)
    scoring = sorted(set(start_step + round((total - start_step) * i / checks)
                         for i in range(1, checks + 1)))
    if parent is not None:
        # Reuse real prefix observations nearest below desired display steps.
        # Every original scoring check remains present; only media selection changes.
        media = sorted(set([start_step, *[
            int(prior_data["steps"][np.searchsorted(prior_data["steps"], step, side="right") - 1])
            if step <= start_step else step for step in media]]))
        scoring = sorted(set(parent["scoring_steps"] + scoring))
    sources = [*sorted((ROOT / "particlegan").rglob("*.py")),
               ROOT / "lib/toy_models.py", ROOT / "lib/denoising_toy.py", ROOT / "lib/toy_metrics.py",
               Path(__file__), Path(__file__).with_name("generate.py")]
    prior_desc = (
        f"MoGParticlePrior(num_particles={prior_spec['num_particles']}, z_dim={prior_spec['z_dim']}, "
        f"sigma={prior_spec['sigma']}, standardize={prior_spec['standardize']}); "
        "public deterministic initialization (matches Forge grid100 task prior)"
        if prior_spec["kind"] == "mog"
        else "recipe.make_prior(); public deterministic initialization"
    )
    plotted_n = min(PLOT_SAMPLES, recipe.num_particles)
    contract = dict(
        schema=1, scope="README mode-acquisition illustration; not Forge qualification",
        cohort=f"{default.name}_{prior_spec['kind']}_{prior_spec['num_particles']}",
        seed=SEED, default_recipe=default.name, recipe=recipe.to_dict(), prior=prior_spec,
        execution_budget=total,
        version=tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"],
        task=dict(target="uniform 10x10 Gaussian grid", sigma=.03,
                  generator="SimpleMLPGenerator(width=128, depth=3)",
                  critic="SimpleMLPDiscriminator(width=128, depth=3, fourier=2)",
                  prior=prior_desc,
                  initialization=dict(generator=0, critic=1, prior=2),
                  sampling=f"live, clean (output_noise=0), uniform prior draws; fixed first {plotted_n} rows for plotting"),
        gate=dict(modes=100, minimum_nearby_samples=10, radius=.09, hq_min=.95,
                  terminal_checks=5, evaluation_samples=EVAL_SAMPLES, full_default_budget=default.total_steps),
        scoring_steps=scoring, media_steps=media, stream_seeds=STREAM_SEEDS,
        controls=dict(oracle=oracle, collapsed=collapsed),
        initial_sha256=initial_hashes,
        source_sha256={str(path.relative_to(ROOT)): digest(path) for path in sources},
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        device=str(device), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    if parent is not None:
        contract["continuation"] = dict(
            parent_directory=str(parent_path), parent_summary_sha256=digest(parent_path / "summary.json"),
            parent_observations_sha256=parent["observations_sha256"],
            parent_checkpoint_sha256=parent["final_checkpoint_sha256"],
            start_step=start_step, parent_status=parent["status"], prefix_updates_repeated=0,
            recipe_unchanged=True, original_scoring_checks_retained=True)
    (out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")

    def checkpoint(name):
        torch.save(dict(trainer=trainer.state_dict(), data_rng=data.get_state(),
                        constructor_rng=trainer.readme_constructor_rng,
                        reference_rng=reference.get_state(), evaluation_initial=evaluation_initial,
                        contract=contract), out / name)

    checkpoint("initial.pt")
    print(json.dumps(dict(event="start", output_dir=str(out), recipe=recipe.name,
                          total_steps=total, device=str(device), controls=contract["controls"])), flush=True)
    observations, coordinates = [], []
    if parent is not None:
        for i, step in enumerate(prior_data["steps"]):
            observations.append(dict(step=int(step), modes=int(prior_data["modes"][i]),
                hq=float(prior_data["hq"][i]), mode_tv=float(prior_data["mode_tv"][i]),
                passed=bool(prior_data["modes"][i] == 100 and prior_data["hq"][i] >= .95),
                scoring=bool(prior_data["scoring"][i]), frame=int(step) in media))
            coordinates.append(prior_data["samples"][i])
        with (out / "metrics.jsonl").open("w") as stream:
            stream.writelines(json.dumps(row) + "\n" for row in observations)
    final_samples = None

    @torch.no_grad()
    def observe(step):
        nonlocal final_samples
        before = {name: getattr(trainer, name).get_state().clone() for name in trainer._STREAMS
                  if name != "eval_generator"}
        trainer.eval_generator.set_state(evaluation_initial)
        samples = trainer.sample(EVAL_SAMPLES)
        plotted = trainer.sample(min(PLOT_SAMPLES, recipe.num_particles), fixed_first_n=True)
        row = dict(step=step, **score(samples, centers), scoring=step in scoring, frame=step in media)
        for name, state in before.items():
            if not torch.equal(state, getattr(trainer, name).get_state()):
                raise RuntimeError(f"observation consumed training stream {name}")
        observations.append(row)
        coordinates.append(plotted.cpu().numpy())
        with (out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if step == total:
            final_samples = samples

    if parent is None:
        observe(0)
    wanted = set(media + scoring)
    train_seconds = 0.
    for step in range(start_step + 1, total + 1):
        tick = time.perf_counter()
        stats = trainer.step(sample_100gaussians(recipe.batch_size, device, generator=data),
                             generator_real=lambda: sample_100gaussians(recipe.batch_size, device, generator=data))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds += time.perf_counter() - tick
        if not all(bool(torch.isfinite(stats[key])) for key in ("loss_d", "loss_g", "penalty")):
            checkpoint("nonfinite.pt")
            raise ValueError(f"nonfinite training at step {step}")
        if step in wanted:
            observe(step)
        if step % 100 == 0 or step == total:
            print(json.dumps(dict(event="progress", step=step, total_steps=total,
                                  loss_d=float(stats["loss_d"]), loss_g=float(stats["loss_g"]),
                                  train_seconds=round(train_seconds, 2), observation=observations[-1])), flush=True)
    checkpoint("final.pt")
    distribution = grid_metrics(final_samples, torch.zeros(EVAL_SAMPLES, device=device, dtype=torch.long),
                                GaussianGrid(device, .03, 1), target)
    checked = [row for row in observations if row["scoring"]]
    passed = total >= (default.total_steps or 1) and len(checked) >= 5 and all(row["passed"] for row in checked[-5:])
    np.savez_compressed(out / "observations.npz", samples=np.stack(coordinates),
                        steps=np.array([row["step"] for row in observations]),
                        hq=np.array([row["hq"] for row in observations]),
                        modes=np.array([row["modes"] for row in observations]),
                        mode_tv=np.array([row["mode_tv"] for row in observations]),
                        scoring=np.array([row["scoring"] for row in observations]),
                        frame=np.array([row["frame"] for row in observations]), centers=centers.cpu().numpy(),
                        target=target.cpu().numpy(), final_evaluation=final_samples.cpu().numpy())
    summary = dict(**contract, completed_steps=trainer.completed_steps, status="PASS" if passed else "FAIL",
                   final=observations[-1], terminal_checks=checked[-5:], distribution_metrics=distribution,
                   train_seconds=train_seconds + (parent["train_seconds"] if parent else 0),
                   new_updates=total - start_step, new_train_seconds=train_seconds,
                   record_seconds=time.perf_counter() - started,
                   observations_sha256=digest(out / "observations.npz"),
                   final_checkpoint_sha256=digest(out / "final.pt"),
                   final_data_rng_sha256=hashlib.sha256(data.get_state().numpy().tobytes()).hexdigest(),
                   observation_training_rng_unchanged=True,
                   limitations="One fixed-seed illustration of mode acquisition; clean particle outputs; "
                                "coverage does not establish Gaussian width/covariance fidelity or a trainer comparison.")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(event="record_complete", status=summary["status"], final=summary["final"],
                          train_seconds=train_seconds, output_dir=str(out))), flush=True)
    if not args.record_only:
        from render import render
        render(out, fps=args.fps, hold=args.hold_seconds)
    return 0 if passed else 1
