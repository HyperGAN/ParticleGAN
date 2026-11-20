"""Render the original ring6400 public-trainer observations; add no draws or updates."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(directory, output):
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle
    from PIL import Image
    from experiments.forge.contracts import stable_hash

    publication = json.loads((Path(__file__).parent / "results.json").read_text())
    recorded = next(r for r in publication["records"]
                    if r["group"] == "k3p" and r["task"] == "ring16_acquisition_6400")
    assert directory.name == recorded["attempt_id"]
    for name, digest in recorded["receipt_sha256"].items():
        assert sha256(directory / name) == digest, name
    raw = json.loads((directory / "raw-result.json").read_text())
    graded = json.loads((directory / "graded-result.json").read_text())
    assert stable_hash(raw) == graded["raw_hash"]
    saved = raw["evidence"]["saved_observer_outputs"]
    path = directory / saved["path"]
    assert sha256(path) == saved["sha256"] and path.stat().st_size == saved["bytes"]
    frames = torch.load(path, weights_only=True, map_location="cpu")
    observations = raw["evidence"]["observations"]
    assert len(frames) == len(observations) == 384
    assert [r["step"] for r in frames] == [r["step"] for r in observations]
    source = recorded["source_commit"]
    frozen = subprocess.check_output(["git", "show", f"{source}:configs/forge/tasks/ring16_acquisition_6400.json"], cwd=root)
    task = json.loads(frozen)
    thresholds = task["evaluation"]["thresholds"]
    means = np.asarray(task["execution"]["host_definition"]["means"])
    chosen = sorted(set(range(0, 384, 8)) | set(range(379, 384)))
    images = []
    for index in chosen:
        row = observations[index]
        assert all(isinstance(row[t[0]], (float, int)) for t in thresholds)
        passing = all(row[k] >= bound if op == ">=" else row[k] <= bound for k, op, bound in thresholds)
        points = frames[index]["samples"].numpy()[:1024]
        fig = Figure(figsize=(8, 4), dpi=80)
        FigureCanvasAgg(fig)
        left, right = fig.subplots(1, 2)
        left.scatter(points[:, 0], points[:, 1], s=2, alpha=.5, color="#2474ab")
        for mean in means:
            left.add_patch(Circle(mean, .2, fill=False, color="#d66d22", linewidth=1))
        left.set(xlim=(-4.5, 4.5), ylim=(-4.5, 4.5), aspect="equal", title="Saved public-prior samples (first 1,024)")
        right.plot([r["step"] for r in observations[:index + 1]],
                   [r["component_min_eigen_ratio"] for r in observations[:index + 1]], linewidth=1)
        right.axhline(.15, color="#d66d22", linestyle="--", label="spread gate: 0.15")
        right.set(xlim=(0, 6400), ylim=(0, 1), xlabel="Update", title="Minimum component spread ratio")
        right.legend(fontsize=8)
        fig.suptitle(f"Historical K3P v2 / ring6400 / step {row['step']} / all gates: {'PASS' if passing else 'FAIL'}")
        fig.tight_layout()
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("P", palette=Image.Palette.ADAPTIVE))
        fig.clear()
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output, save_all=True, append_images=images[1:], duration=150, loop=0)
    receipt = {"schema": "archived_ring6400_training_gif_v1", "qualification_input": False,
               "kind": "actual_training_saved_observations_gif", "optimizer_updates_added": 0, "sampling_draws_added": 0,
               "source_commit": source, "candidate_id": recorded["candidate_id"],
               "candidate_revision": recorded["candidate_revision"], "attempt_id": recorded["attempt_id"],
               "task": recorded["task"], "recorded_grade": recorded["status"], "seed": recorded["seed"],
               "raw_sha256": sha256(directory / "raw-result.json"), "saved_observer_outputs": saved,
               "frozen_task_sha256": hashlib.sha256(frozen).hexdigest(), "frames": len(chosen),
               "frame_indices": chosen, "samples_displayed_per_frame": 1024,
               "gif_sha256": sha256(output), "scope": "Original draft10/source cohort; no current-view17 qualification."}
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"Rendered {len(chosen)} original observation frames; zero updates and zero sampling draws: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("ring6400-training.gif"))
    args = parser.parse_args()
    render(args.directory, args.output)
