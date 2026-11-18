"""Render measured transport and quality gaps from the local paired reduction."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def load(directory, task, kind):
    with (directory / f"{task}-{kind}.jsonl").open() as source:
        return [json.loads(line) for line in source]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 5.9), sharex="col", layout="constrained")
    for col, (task, title, metric, threshold) in enumerate((
            ("vector_unequal_mass", "Unequal-mass Gaussians", "width", .05),
            ("interp_sparse_arc", "Sparse arc", "mean_arc_error", .03))):
        plans = load(args.traces, task, "plans")
        quality = load(args.traces, task, "quality")
        top, bottom = axes[:, col]
        x = [r["update"] for r in plans]
        top.plot(x, [100 * r["between_tv"] for r in plans], color="#17608a", lw=1.0, label="Executed plans, between arms")
        top.plot(x, [100 * r["direct_tv"] for r in plans], color="#ab5520", lw=1.0, label="Metric switch, same feature-arm state")
        top.axhline(1, ls="--", color="0.32", lw=1.0, label="1% threshold")
        top.set(yscale="symlog", ylim=(-.0007, 120), title=title)
        top.set_yscale("symlog", linthresh=.001)
        top.set_yticks([0, .01, .1, 1, 10, 100])
        top.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        if col == 0:
            top.set_ylabel("Transport mass moved (%)")
        gap = [abs(row["parent"][metric] - row["feature"][metric])
               if None not in (row["parent"][metric], row["feature"][metric]) else np.nan for row in quality]
        bottom.plot([row["step"] for row in quality], gap, color="#365d49", lw=1.0, marker=".", ms=3)
        bottom.axhline(threshold, color="0.32", ls="--", lw=1.0)
        bottom.set_ylabel("Absolute rare-width gap" if col == 0 else "Absolute mean arc-error gap")
        bottom.set_xlabel("Training update")
        bottom.set_ylim(bottom=0)
        bottom.set_xscale("symlog", linthresh=20)
        ticks = [0, 10, 30, 100, 300, x[-1]]
        bottom.set_xticks(ticks, [str(t) for t in ticks])
        bottom.set_xlim(0, x[-1])
        for ax in (top, bottom):
            ax.grid(axis="y", lw=.5, alpha=.27)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(loc="lower right", fontsize=8, framealpha=.95)
    fig.savefig(args.output, dpi=160, facecolor="white")


if __name__ == "__main__":
    main()
