import json,os,math
import pandas as pd
import numpy as np
from scipy.stats import t as student_t

from .global_sweep import run_global_budget_sweep,parse_weight_profiles

def load_manifest(path):
    with open(path) as f:
        data=json.load(f)
    if not isinstance(data,list):
        raise ValueError("manifest must be a JSON list of replicate objects")
    return data

def run_replicates(manifest_path,output_root,budgets,targets,epoch_sizes,weight_profiles):
    """
    Manifest format:
    [
      {
        "name":"2019q1",
        "trip_total":"results/range-2019q1/chicago-total/stream.csv",
        "trip_miles":"results/range-2019q1/chicago-miles/stream.csv",
        "trip_seconds":"results/range-2019q1/chicago-seconds/stream.csv"
      }
    ]
    """
    os.makedirs(output_root,exist_ok=True)
    manifest=load_manifest(manifest_path)
    rows=[]

    for item in manifest:
        name=item["name"]
        streams={
            "trip_total":item["trip_total"],
            "trip_miles":item["trip_miles"],
            "trip_seconds":item["trip_seconds"],
        }
        out=os.path.join(output_root,name)
        df=run_global_budget_sweep(
            streams,out,budgets,targets,epoch_sizes,weight_profiles
        )
        df.insert(0,"replicate",name)
        rows.append(df)

    combined=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    combined.to_csv(os.path.join(output_root,"replicate_results.csv"),index=False)
    return combined

def _ci95(series):
    """Two-sided 95% Student-t interval over independent workload replicates."""
    x=pd.Series(series).dropna().astype(float)
    n=len(x)
    if n==0:
        return (None,None,None,None)
    mean=float(x.mean())
    sd=float(x.std(ddof=1)) if n>1 else 0.0
    if n>1:
        critical=float(student_t.ppf(0.975,df=n-1))
        half=critical*sd/math.sqrt(n)
    else:
        half=0.0
    return mean,sd,mean-half,mean+half

def aggregate_replicates(replicate_csv,output_dir):
    os.makedirs(output_dir,exist_ok=True)
    df=pd.read_csv(replicate_csv)

    keys=["epoch_size","target","budget","weight_profile"]
    rows=[]

    for key,g in df.groupby(keys):
        feasible=g[g["feasible"]==True].copy()
        n_total=len(g)
        n_feasible=len(feasible)

        err_mean,err_sd,err_lo,err_hi=_ci95(
            feasible["error_improvement_fraction"]
        )
        hit_mean,hit_sd,hit_lo,hit_hi=_ci95(
            feasible["target_hit_rate_gain"]
        )
        adaptive_err_mean,adaptive_err_sd,adaptive_err_lo,adaptive_err_hi=_ci95(
            feasible["adaptive_weighted_mean_error"]
        )

        dominant=feasible[
            (feasible["error_improvement_fraction"]>0) &
            (feasible["target_hit_rate_gain"]>=0)
        ]

        rows.append({
            "epoch_size":key[0],
            "target":key[1],
            "budget":key[2],
            "weight_profile":key[3],
            "replicates_total":n_total,
            "replicates_feasible":n_feasible,
            "feasible_fraction":n_feasible/max(n_total,1),
            "dominant_replicates":len(dominant),
            "dominance_rate":len(dominant)/max(n_feasible,1),
            "ci_method":"student_t_95_two_sided",
            "mean_error_improvement_fraction":err_mean,
            "sd_error_improvement_fraction":err_sd,
            "ci95_error_improvement_low":err_lo,
            "ci95_error_improvement_high":err_hi,
            "mean_target_hit_rate_gain":hit_mean,
            "sd_target_hit_rate_gain":hit_sd,
            "ci95_target_hit_rate_gain_low":hit_lo,
            "ci95_target_hit_rate_gain_high":hit_hi,
            "mean_adaptive_weighted_error":adaptive_err_mean,
            "ci95_adaptive_weighted_error_low":adaptive_err_lo,
            "ci95_adaptive_weighted_error_high":adaptive_err_hi,
        })

    out=pd.DataFrame(rows)
    out.to_csv(os.path.join(output_dir,"aggregate_summary.csv"),index=False)

    strong=out[
        (out["replicates_feasible"]>=5) &
        (out["dominance_rate"]>=0.8) &
        (out["ci95_error_improvement_low"]>0)
    ].copy()

    strong.sort_values(
        ["mean_error_improvement_fraction","dominance_rate"],
        ascending=False
    ).to_csv(os.path.join(output_dir,"publication_candidates.csv"),index=False)

    return out
