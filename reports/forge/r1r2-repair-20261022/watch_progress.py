"""Render the local append-only queue event log as concise, tail-able progress.

This observes events only. It never submits, grades or alters a run.
"""
import argparse
import json
from pathlib import Path
import time

METRICS = ("component_covariance_error", "component_min_eigen_ratio", "hq", "modes", "mass_tv",
           "minimum_reconstruction_token_probability", "final_projection_cdf", "paired_nmse")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("--idle-seconds", type=float, default=120)
    args = parser.parse_args()
    last = time.monotonic()
    with args.events.open() as stream:
        while time.monotonic() - last < args.idle_seconds:
            line = stream.readline()
            if not line:
                time.sleep(.5)
                continue
            last = time.monotonic()
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("campaign") != "r1r2-repair-20261022":
                continue
            event = row.get("event", "")
            if not event.startswith("worker_"):
                continue
            role = row.get("candidate", "").removeprefix("r1r2-repair-").removesuffix("-20261022")
            metrics = row.get("metrics", {})
            cells = " ".join(f"{key}={metrics[key]:.6g}" for key in METRICS
                             if isinstance(metrics.get(key), (int, float)))
            print(f"{row.get('timestamp', '')} {role} {row.get('task', '')} {event} "
                  f"step={row.get('step', '-')} {row.get('status', '')} {cells}", flush=True)


if __name__ == "__main__":
    main()
