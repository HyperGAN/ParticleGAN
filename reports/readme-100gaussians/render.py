"""Render the README GIF from saved actual observations, without importing Torch.

Usage: python reports/readme-100gaussians/render.py /path/to/recorded-run
A source-bound NPZ and receipt are required. No training or interpolation occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np
from PIL import Image

SIZE = (960, 640)
BG, PANEL, INK, MUTED = "#08121e", "#0c1b2b", "#edf5ff", "#91a7bf"
CYAN, GOLD, GREEN, RED = "#61dbff", "#ffcd86", "#89e6ba", "#ffb2a6"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_run(directory):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    if sha256(directory / "observations.npz") != summary["observations_sha256"]:
        raise ValueError("observations do not match the recorded training receipt")
    with np.load(directory / "observations.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    steps = data["steps"]
    if (len(steps) < 2 or steps[0] != 0 or np.any(np.diff(steps) <= 0)
            or steps[-1] != summary["completed_steps"]):
        raise ValueError("incomplete or unordered observed training steps")
    if not np.isfinite(data["samples"]).all():
        raise ValueError("nonfinite plotted observations")
    if steps[data["frame"]].tolist() != summary["media_steps"]:
        raise ValueError("media schedule differs from the frozen contract")
    if steps[data["scoring"]].tolist() != summary["scoring_steps"]:
        raise ValueError("scoring schedule differs from the frozen contract")
    if not (data["hq"][-1] == summary["final"]["hq"]
            and data["modes"][-1] == summary["final"]["modes"]):
        raise ValueError("final caption metrics differ from the receipt")
    passed = ((data["modes"] == summary["gate"]["modes"])
              & (data["hq"] >= summary["gate"]["hq_min"]))
    terminal = passed[data["scoring"]][-summary["gate"]["terminal_checks"]:]
    full = (summary["completed_steps"] == summary.get("execution_budget", summary["gate"]["full_default_budget"])
            and summary["completed_steps"] >= summary["gate"]["full_default_budget"]
            and len(terminal) == summary["gate"]["terminal_checks"] and bool(terminal.all()))
    if summary["status"] != ("PASS" if full else "FAIL"):
        raise ValueError("gate verdict differs from the fixed numerical checks")
    return summary, data


class TrainingFigure:
    def __init__(self, summary, data):
        self.summary, self.data = summary, data
        self.fig = plt.figure(figsize=(SIZE[0] / 100, SIZE[1] / 100), dpi=100, facecolor=BG)
        self.fig.text(.055, .924, "ParticleGAN", color=INK, fontsize=25, weight="bold")
        self.fig.text(.057, .875, "100 Gaussians  /  learning a particle prior", color=MUTED, fontsize=11)
        recipe = summary["recipe"]["name"]
        title = "Optimized BISPF" if recipe == "bispf_optimized" else recipe.replace("_", " ").upper()
        self.fig.text(.944, .94, "v" + summary["version"], color=MUTED, fontsize=9, ha="right")
        self.fig.text(.944, .895, title, color=CYAN, fontsize=12, ha="right", weight="bold")
        self.ax = self.fig.add_axes((.055, .125, .555, .70), facecolor=PANEL)
        self.ax.set(xlim=(-5.6, 5.6), ylim=(-5.6, 5.6), aspect="equal", xticks=[], yticks=[])
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#24364b")
        for value in np.arange(-5, 6):
            self.ax.axvline(value, color="#15283b", linewidth=.45, zorder=0)
            self.ax.axhline(value, color="#15283b", linewidth=.45, zorder=0)
        self.ax.add_collection(PatchCollection(
            [Circle(center, .09) for center in data["centers"]],
            facecolors="none", edgecolors=GOLD, linewidths=.65, alpha=.65, zorder=1))
        self.trails = LineCollection([], colors=CYAN, alpha=.10, linewidths=.45, zorder=2)
        self.ax.add_collection(self.trails)
        n = data["samples"].shape[1]
        self.glow = self.ax.scatter(np.zeros(n), np.zeros(n), s=15, color=CYAN, alpha=.035,
                                    linewidths=0, zorder=3)
        self.points = self.ax.scatter(np.zeros(n), np.zeros(n), s=1.8, color=CYAN, alpha=.8,
                                      linewidths=0, zorder=4)
        self.fig.text(.068, .826, "MODEL OUTPUTS", color=MUTED, fontsize=8, weight="bold")
        self.fig.text(.59, .826, "fixed axes", color=MUTED, fontsize=8, ha="right")
        self.fig.text(.666, .785, "MODES COVERED", color=MUTED, fontsize=9, weight="bold")
        self.mode_text = self.fig.text(.665, .708, "", color=INK, fontsize=32, weight="bold")
        self.fig.text(.666, .655, "SAMPLES WITHIN 3σ", color=MUTED, fontsize=9, weight="bold")
        self.hq_text = self.fig.text(.665, .583, "", color=CYAN, fontsize=30, weight="bold")
        self.fig.text(.666, .531, "goal: 100 modes · at least 95%", color=MUTED, fontsize=9)
        self.gate_text = self.fig.text(.666, .485, "", color=MUTED, fontsize=10, weight="bold")
        self.chart = self.fig.add_axes((.677, .252, .265, .157), facecolor=BG)
        self.chart.set(xlim=(0, summary["completed_steps"]), ylim=(-3, 103),
                       yticks=[0, 50, 100], xticks=[0, summary["completed_steps"]])
        self.chart.tick_params(colors=MUTED, labelsize=8, length=0, pad=5)
        for side, spine in self.chart.spines.items():
            spine.set_visible(side in ("left", "bottom"))
            spine.set_edgecolor("#304358")
        self.chart.grid(axis="y", color="#203146", linewidth=.5)
        self.coverage_line, = self.chart.plot([], [], color=GOLD, linewidth=1.4)
        self.hq_line, = self.chart.plot([], [], color=CYAN, linewidth=1.6)
        self.fig.text(.677, .44, "OBSERVED THROUGH TRAINING", color=MUTED, fontsize=8, weight="bold")
        self.fig.text(.677, .203, "— modes / 100", color=GOLD, fontsize=8)
        self.fig.text(.83, .203, "— within 3σ", color=CYAN, fontsize=8)
        self.step_text = self.fig.text(.666, .136, "", color=INK, fontsize=13, weight="bold")
        self.fig.text(.066, .089, "○ target: 3σ radius", color=GOLD, fontsize=8.5)
        self.fig.text(.292, .089, "● learned particle outputs", color=CYAN, fontsize=8.5)
        prior_info = summary.get("prior") or {}
        footer = (
            f"ACTUAL TRAINING  ·  live weights  ·  MoG prior (σ={prior_info['sigma']:g})  ·  clean  ·  seed 0"
            if prior_info.get("kind") == "mog"
            else "ACTUAL TRAINING  ·  live weights  ·  clean samples  ·  seed 0"
        )
        self.fig.text(.055, .031, footer, color=MUTED, fontsize=8)
        self.fig.text(.945, .031, f'{n:,} fixed row IDs  /  20,000 evaluation draws',
                      ha="right", color=MUTED, fontsize=8)
        self.progress_bg = FancyBboxPatch((.666, .10), .276, .005, boxstyle="square,pad=0",
                                         transform=self.fig.transFigure, facecolor="#23384e", edgecolor="none")
        self.fig.patches.append(self.progress_bg)
        self.progress = FancyBboxPatch((.666, .10), 0, .005, boxstyle="square,pad=0",
                                      transform=self.fig.transFigure, facecolor=CYAN, edgecolor="none")
        self.fig.patches.append(self.progress)

    def frame(self, index, recent):
        data = self.data
        step, total = int(data["steps"][index]), self.summary["completed_steps"]
        points = data["samples"][index]
        self.points.set_offsets(points)
        self.glow.set_offsets(points)
        # Straight segments connect only saved observations of the same row IDs.
        path = data["samples"][recent, ::16]
        self.trails.set_segments(np.transpose(path, (1, 0, 2)) if len(recent) > 1 else [])
        self.mode_text.set_text(f'{int(data["modes"][index])} / 100')
        self.hq_text.set_text(f'{data["hq"][index]:.1%}')
        if step == total:
            status = self.summary["status"]
            self.gate_text.set_text(f"FINAL GOAL: {status}")
            self.gate_text.set_color(GREEN if status == "PASS" else RED)
        else:
            passed = data["modes"][index] == 100 and data["hq"][index] >= .95
            self.gate_text.set_text("CURRENT GOAL: MET" if passed else "CURRENT GOAL: NOT MET")
            self.gate_text.set_color(GREEN if passed else MUTED)
        self.coverage_line.set_data(data["steps"][:index + 1], data["modes"][:index + 1])
        self.hq_line.set_data(data["steps"][:index + 1], 100 * data["hq"][:index + 1])
        self.step_text.set_text(f"UPDATE {step:,} / {total:,}")
        self.progress.set_width(.276 * step / total)
        self.fig.canvas.draw()
        return Image.frombuffer("RGBA", SIZE, self.fig.canvas.buffer_rgba()).convert("RGB")

    def close(self):
        plt.close(self.fig)


def render(directory, *, fps=25, hold=2.5):
    directory = Path(directory)
    if fps <= 0 or 100 % fps or not np.isfinite(hold) or hold < 0:
        raise ValueError("fps must divide 100; hold must be finite and nonnegative")
    summary, data = read_run(directory)
    indices = np.flatnonzero(data["frame"])
    figure = TrainingFigure(summary, data)
    try:
        frames = [figure.frame(int(index), indices[max(0, i - 2):i + 1])
                  for i, index in enumerate(indices)]
    finally:
        figure.close()
    # One shared palette prevents flicker. Reserve consistent colors for every frame.
    picks = frames[::max(1, len(frames) // 20)]
    strip = Image.new("RGB", (SIZE[0], SIZE[1] * len(picks)))
    for i, frame in enumerate(picks):
        strip.paste(frame, (0, SIZE[1] * i))
    palette = strip.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    frame_ms, hold_ms = 1000 // fps, max(round(hold * 100) * 10, 1000 // fps)
    output = directory / "100gaussians-live.gif"
    indexed[0].save(output, save_all=True, append_images=indexed[1:],
                    duration=[frame_ms] * (len(indexed) - 1) + [hold_ms], loop=0, optimize=True)
    frames[0].save(directory / "preview-start.png")
    frames[-1].save(directory / "preview-final.png")
    receipt = dict(gif_sha256=sha256(output), bytes=output.stat().st_size, size=list(SIZE),
                   frames=len(frames), frame_steps=data["steps"][indices].tolist(),
                   frame_ms=frame_ms, final_hold_ms=hold_ms, renderer_sha256=sha256(__file__),
                   observations_sha256=summary["observations_sha256"],
                   training_summary_sha256=sha256(directory / "summary.json"), status=summary["status"])
    (directory / "render.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(event="render_complete", output=str(output), **receipt)), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--hold-seconds", type=float, default=2.5)
    args = parser.parse_args()
    render(args.directory, fps=args.fps, hold=args.hold_seconds)
