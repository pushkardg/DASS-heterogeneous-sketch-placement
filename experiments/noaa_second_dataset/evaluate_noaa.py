#!/usr/bin/env python3
"""Profile NOAA streams and evaluate HASP, LEF, and best feasible uniform K.

All reported values are computed from the extracted NOAA files. No expected
improvement or paper result is encoded in this script.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dass.allocators import greedy_benefit_per_byte, largest_error_first
from dass.evaluator import evaluate_rank_error
from dass.sketch import build_kll, serialized_bytes

KS = (64, 128, 256, 512, 1024)
METRICS = ("tmax", "prcp", "awnd")
WEIGHT_PROFILES = {
    "uniform": {"tmax": 1.0, "prcp": 1.0, "awnd": 1.0},
    "tmax_heavy": {"tmax": 4.0, "prcp": 1.0, "awnd": 1.0},
    "awnd_heavy": {"tmax": 1.0, "prcp": 1.0, "awnd": 4.0},
}


def profile_file(path, stream, epoch_size):
    values = np.loadtxt(path, dtype=float)
    rows = []
    epoch = 0
    for start in range(0, len(values), epoch_size):
        part = values[start:start + epoch_size]
        if len(part) < 100:
            continue
        for k in KS:
            t0 = time.perf_counter()
            sk = build_kll(part, k)
            ev = evaluate_rank_error(sk, part)
            rows.append({
                "stream": stream, "epoch": epoch, "k": k,
                "bytes": serialized_bytes(sk),
                "max_rank_error": ev["max_rank_error"],
                "build_wall_ms": (time.perf_counter() - t0) * 1000.0,
            })
        epoch += 1
    return rows


def metric_of(stream):
    return stream.rsplit("__", 1)[-1]


def weights_for(streams, profile):
    return {s: float(profile[metric_of(s)]) for s in streams}


def weighted_uniform_error(profiles, k, weights):
    x = profiles[profiles.k == k]
    numerator = 0.0
    denominator = 0.0
    for r in x.itertuples():
        w = float(weights[r.stream])
        numerator += w * float(r.max_rank_error)
        denominator += w
    return numerator / denominator


def uniform_footprints(profiles):
    return {int(k): int(profiles[profiles.k == k].bytes.sum()) for k in KS}


def choose_bpu(profiles):
    """Choose three data-derived budgets between adjacent uniform tiers.

    We use average bytes/unit so the same budget rule can be applied to every
    yearly slice without importing Chicago quotas.
    """
    units = profiles[["stream", "epoch"]].drop_duplicates().shape[0]
    fps = uniform_footprints(profiles)
    per_unit = {k: fps[k] / units for k in KS}
    mids = []
    for a, b in ((64, 128), (128, 256), (256, 512)):
        mids.append(int(round((per_unit[a] + per_unit[b]) / 2.0)))
    return mids, per_unit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="results/ieee-bigdata-2026/noaa/data")
    p.add_argument("--output", default="results/ieee-bigdata-2026/noaa/evaluation")
    p.add_argument("--years", nargs="+", type=int, default=[2019, 2020, 2021, 2022, 2023])
    p.add_argument("--epoch-size", type=int, default=100000)
    p.add_argument("--target", type=float, default=0.005)
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    started = time.perf_counter()

    for year in args.years:
        for metric in METRICS:
            path = Path(args.data_root) / str(year) / f"noaa_{year}_{metric}.txt"
            if not path.exists():
                raise FileNotFoundError(path)
            stream = f"{year}__{metric}"
            print(f"Profiling {stream}: {path}", flush=True)
            all_rows.extend(profile_file(path, stream, args.epoch_size))

    profiles = pd.DataFrame(all_rows)
    profiles.to_csv(out / "candidate_profiles.csv", index=False)

    # Derive budgets only from measured NOAA footprints, never Chicago quotas.
    bpus, overall_per_unit = choose_bpu(profiles)
    footprint_rows = []
    for year in args.years:
        yp = profiles[profiles.stream.str.startswith(f"{year}__")]
        units = yp[["stream", "epoch"]].drop_duplicates().shape[0]
        fps = uniform_footprints(yp)
        for k in KS:
            footprint_rows.append({"year": year, "units": units, "k": k,
                                   "uniform_bytes": fps[k],
                                   "bytes_per_unit": fps[k] / units})
    pd.DataFrame(footprint_rows).to_csv(out / "uniform_tier_footprints.csv", index=False)

    results = []
    for year in args.years:
        yp = profiles[profiles.stream.str.startswith(f"{year}__")].copy()
        streams = sorted(yp.stream.unique())
        units = yp[["stream", "epoch"]].drop_duplicates().shape[0]
        fps = uniform_footprints(yp)
        min_required = fps[64]
        for bpu in bpus:
            budget = int(bpu * units)
            for wp_name, wp in WEIGHT_PROFILES.items():
                weights = weights_for(streams, wp)
                halloc, hm = greedy_benefit_per_byte(yp, budget, args.target, weights)
                lalloc, lm = largest_error_first(yp, budget, args.target, weights)
                feasible_uniform = [k for k in KS if fps[k] <= budget]
                if not feasible_uniform:
                    results.append({"year": year, "units": units, "bytes_per_unit": bpu,
                                    "budget": budget, "weight_profile": wp_name,
                                    "feasible": False, "minimum_required_bytes": min_required})
                    continue
                uk = max(feasible_uniform)
                ue = weighted_uniform_error(yp, uk, weights)
                he = hm["weighted_mean_error"] if hm.get("within_budget") else np.nan
                le = lm["weighted_mean_error"] if lm.get("within_budget") else np.nan
                results.append({
                    "year": year, "units": units, "bytes_per_unit": bpu, "budget": budget,
                    "weight_profile": wp_name, "feasible": True,
                    "uniform_k": uk, "uniform_error": ue,
                    "hasp_error": he, "lef_error": le,
                    "hasp_used_bytes": hm.get("used_bytes"), "lef_used_bytes": lm.get("used_bytes"),
                    "hasp_runtime_ms": hm.get("runtime_ms"), "lef_runtime_ms": lm.get("runtime_ms"),
                    "hasp_improvement_vs_uniform_pct": (ue - he) / ue * 100.0 if ue else 0.0,
                    "lef_improvement_vs_uniform_pct": (ue - le) / ue * 100.0 if ue else 0.0,
                    "hasp_improvement_vs_lef_pct": (le - he) / le * 100.0 if le else 0.0,
                })

    rdf = pd.DataFrame(results)
    rdf.to_csv(out / "per_configuration_results.csv", index=False)
    feasible = rdf[rdf.feasible == True].copy()  # noqa: E712
    summary = {
        "dataset": "NOAA GHCN-Daily",
        "years": args.years,
        "metrics": list(METRICS),
        "epoch_size": args.epoch_size,
        "candidate_ks": list(KS),
        "target": args.target,
        "budget_rule": "three midpoint bytes/unit values between measured NOAA uniform K=64/128, 128/256, and 256/512 tiers",
        "selected_bytes_per_unit": bpus,
        "overall_uniform_bytes_per_unit": {str(k): overall_per_unit[k] for k in KS},
        "profile_rows": int(len(profiles)),
        "repository_units": int(profiles[["stream", "epoch"]].drop_duplicates().shape[0]),
        "configurations": int(len(rdf)),
        "feasible_configurations": int(len(feasible)),
        "hasp_wins_vs_uniform": int((feasible.hasp_improvement_vs_uniform_pct > 0).sum()),
        "hasp_losses_vs_uniform": int((feasible.hasp_improvement_vs_uniform_pct < 0).sum()),
        "hasp_wins_vs_lef": int((feasible.hasp_improvement_vs_lef_pct > 0).sum()),
        "lef_wins_vs_hasp": int((feasible.hasp_improvement_vs_lef_pct < 0).sum()),
        "mean_hasp_improvement_vs_uniform_pct": float(feasible.hasp_improvement_vs_uniform_pct.mean()) if len(feasible) else None,
        "median_hasp_improvement_vs_uniform_pct": float(feasible.hasp_improvement_vs_uniform_pct.median()) if len(feasible) else None,
        "mean_hasp_improvement_vs_lef_pct": float(feasible.hasp_improvement_vs_lef_pct.mean()) if len(feasible) else None,
        "wall_seconds": time.perf_counter() - started,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "methodological_scope": "cross-domain validation on one additional public domain; not a claim of universal generality",
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out / "command_manifest.txt", "w") as f:
        f.write("python experiments/noaa_second_dataset/evaluate_noaa.py \\\n")
        f.write(f"  --data-root {args.data_root} --output {args.output} --epoch-size {args.epoch_size} --target {args.target}\n")

    print(json.dumps(summary, indent=2))
    print(f"Wrote measured artifacts to {out}")


if __name__ == "__main__":
    main()
