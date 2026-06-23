"""Recompute TrajectoryMetrics for existing experiment JSONs.

Use when metrics.py has been updated (new metrics added, formulas changed)
but the poll data is still valid. No LLM inference needed — just reruns
compute_trajectory_metrics on the stored polls.

Usage:
    python -m bayesbench.recommender_system.recompute_metrics [--experiments-dir recommender_system/experiments]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import TrajectoryResult
from .data import prepare_data
from .metrics import compute_trajectory_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-dir", type=str,
                        default=str(Path(__file__).parent / "experiments"))
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--n-types", type=int, default=4)
    args = parser.parse_args()

    exp_dir = Path(args.experiments_dir)

    print("Loading TypeModel...")
    type_model, _ = prepare_data(data_dir=args.data_dir, n_types=args.n_types)
    print("TypeModel loaded.\n")

    files = sorted(exp_dir.glob("*.json"))
    files = [f for f in files if ".partial" not in f.name and "_analysis" not in f.name]
    print(f"Recomputing metrics for {len(files)} files...\n")

    ok = 0
    skipped = 0
    errors = 0
    for i, fpath in enumerate(files):
        try:
            with open(fpath) as f:
                data = json.load(f)
            result = TrajectoryResult.from_dict(data)
            if not result.polls:
                skipped += 1
                continue
            result.metrics = compute_trajectory_metrics(result, type_model)
            with open(fpath, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            ok += 1
        except Exception as e:
            print(f"  FAILED {fpath.name}: {e}")
            errors += 1

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)} processed ({ok} ok, {skipped} skipped, {errors} errors)")

    print(f"\nDone: {ok} recomputed, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
