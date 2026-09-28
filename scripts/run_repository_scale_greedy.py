#!/usr/bin/env python3
"""Repository-scale benchmark for the DASS greedy allocation core.

This benchmark intentionally excludes candidate-sketch construction. It creates
synthetic cost/error operating points for N logical summaries, constructs the
same adjacent-upgrade benefit-per-byte scores used by DASS, sorts the upgrades,
and scans them under a proportional byte budget. It reports wall time and peak
RSS for 1K--1M units by default.

The purpose is to measure allocator scaling independently from exact DP and from
raw-data profiling. Results should be described as a synthetic allocator-core
scaling experiment, not as end-to-end storage-system throughput.
"""
import argparse
import csv
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

KS = np.asarray([64, 128, 256, 512, 1024], dtype=np.int32)
# Representative monotone serialized-byte costs. Exact values are not a claim
# about a particular dataset; the benchmark measures allocation-core scaling.
COSTS = np.asarray([1800, 3100, 6000, 11500, 22750], dtype=np.int64)


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return float(r) / (1024.0 if os.uname().sysname == "Linux" else 1024.0 * 1024.0)


def run_once(n: int, seed: int, budget_fraction: float):
    rng = np.random.default_rng(seed)
    start_rss = rss_mb()
    t0 = time.perf_counter()

    # Heterogeneous baseline difficulty and diminishing per-step improvements.
    base_error = rng.lognormal(mean=-5.1, sigma=0.55, size=n)
    weights = rng.choice(np.asarray([1.0, 1.0, 1.0, 4.0]), size=n)
    decay = rng.uniform(0.42, 0.72, size=n)

    # Four adjacent upgrades per unit. Benefit is weighted error reduction;
    # score is the DASS marginal benefit per additional byte.
    steps = len(KS) - 1
    unit_ids = np.repeat(np.arange(n, dtype=np.int32), steps)
    step_ids = np.tile(np.arange(steps, dtype=np.int8), n)
    improvement = np.empty(n * steps, dtype=np.float64)
    for s in range(steps):
        mask = step_ids == s
        improvement[mask] = weights * base_error * (1.0 - decay) * np.power(decay, s)
    delta_bytes = (COSTS[1:] - COSTS[:-1])[step_ids]
    scores = improvement / delta_bytes

    order = np.argsort(scores)[::-1]
    min_bytes = int(n * COSTS[0])
    max_extra = int(n * (COSTS[-1] - COSTS[0]))
    extra_budget = int(max_extra * budget_fraction)
    budget = min_bytes + extra_budget

    # Respect the ordered-upgrade constraint: step s is legal only after s-1.
    current_step = np.zeros(n, dtype=np.int8)
    used = min_bytes
    accepted = 0
    for idx in order:
        u = int(unit_ids[idx]); s = int(step_ids[idx])
        if current_step[u] != s:
            continue
        cost = int(delta_bytes[idx])
        if used + cost <= budget:
            used += cost
            current_step[u] += 1
            accepted += 1

    wall = time.perf_counter() - t0
    return {
        "units": n,
        "candidate_k_values": len(KS),
        "upgrade_candidates": n * steps,
        "budget_bytes": budget,
        "used_bytes": used,
        "accepted_upgrades": accepted,
        "wall_seconds": wall,
        "peak_rss_mb": rss_mb(),
        "rss_growth_mb": max(0.0, rss_mb() - start_rss),
        "seed": seed,
        "budget_fraction": budget_fraction,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit-counts", default="1000,10000,100000,1000000")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--budget-fraction", type=float, default=0.35)
    ap.add_argument("--output", default="results/chicago-v13/repository-scale")
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in [int(x) for x in args.unit_counts.split(",")]:
        for r in range(args.repeats):
            row = run_once(n, args.seed + r, args.budget_fraction)
            row["repeat"] = r
            rows.append(row)
            print(json.dumps(row))

    fields = list(rows[0])
    with (out / "repository_scale_runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    summary = []
    for n in sorted({r["units"] for r in rows}):
        rs = [r for r in rows if r["units"] == n]
        summary.append({
            "units": n,
            "repeats": len(rs),
            "wall_seconds_mean": float(np.mean([r["wall_seconds"] for r in rs])),
            "wall_seconds_p95": float(np.quantile([r["wall_seconds"] for r in rs], .95)),
            "peak_rss_mb_max": float(max(r["peak_rss_mb"] for r in rs)),
            "rss_growth_mb_max": float(max(r["rss_growth_mb"] for r in rs)),
        })
    with (out / "repository_scale_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    with (out / "metadata.json").open("w") as f:
        json.dump({
            "benchmark": "synthetic DASS adjacent-upgrade allocator core",
            "unit_counts": [int(x) for x in args.unit_counts.split(",")],
            "repeats": args.repeats,
            "candidate_k_values": KS.tolist(),
            "note": "Allocator-only benchmark; excludes sketch profiling, catalog I/O, and exact DP."
        }, f, indent=2)


if __name__ == "__main__":
    main()
