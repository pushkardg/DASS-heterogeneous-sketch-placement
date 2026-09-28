import csv,json,os,itertools,math
import numpy as np
import pandas as pd

from .sketch import build_kll,serialized_bytes
from .evaluator import evaluate_rank_error

KS=[64,128,256,512,1024]

OBJECTIVE_NAME="minimize_weighted_mean_max_rank_error_subject_to_storage_budget"

def load_stream(path):
    with open(path) as f:
        return np.asarray([float(r["value"]) for r in csv.DictReader(f)],dtype=float)

def parse_weight_profiles(spec):
    """
    Example:
      uniform:1,1,1;total_heavy:4,1,1;seconds_heavy:1,1,4
    Order is trip_total, trip_miles, trip_seconds.
    """
    out={}
    for profile in spec.split(";"):
        name,vals=profile.split(":",1)
        a,b,c=[float(x) for x in vals.split(",")]
        out[name]={
            "trip_total":a,
            "trip_miles":b,
            "trip_seconds":c,
        }
    return out

def precompute_options(streams,epoch_size,cache_csv=None):
    rows=[]
    for stream_name,path in streams.items():
        arr=load_stream(path)
        for epoch,start in enumerate(range(0,len(arr),epoch_size)):
            x=arr[start:start+epoch_size]
            if len(x)<100:
                continue
            for k in KS:
                s=build_kll(x,k)
                ev=evaluate_rank_error(s,x)
                rows.append({
                    "stream":stream_name,
                    "epoch":epoch,
                    "k":k,
                    "bytes":serialized_bytes(s),
                    "max_rank_error":ev["max_rank_error"],
                })
    df=pd.DataFrame(rows)
    if cache_csv:
        os.makedirs(os.path.dirname(cache_csv) or ".",exist_ok=True)
        df.to_csv(cache_csv,index=False)
    return df

def minimum_required_bytes(options):
    minimum=options[options["k"]==min(KS)]
    return int(minimum["bytes"].sum())

def allocate_from_options(options,total_budget,target,weights):
    """
    Solve the publication objective approximately with greedy marginal upgrades:

      minimize sum_i w_i * E_i(K_i)
      subject to sum_i C_i(K_i) <= B

    The target is NOT part of the allocation objective. It is used only for
    reporting target-hit rate after allocation.
    """
    grouped={
        (name,int(epoch)):g.sort_values("k").reset_index(drop=True)
        for (name,epoch),g in options.groupby(["stream","epoch"])
    }

    min_required=sum(int(g.iloc[0]["bytes"]) for g in grouped.values())
    if total_budget < min_required:
        return (
            pd.DataFrame(),
            {
                "feasible":False,
                "minimum_required_bytes":min_required,
                "used_bytes":0,
                "weighted_mean_error":None,
                "weighted_target_hit_rate":None,
            },
            _uniform_baselines(options,total_budget,target,weights),
        )

    selected={key:0 for key in grouped}
    used=min_required

    upgrades=[]
    for key,g in grouped.items():
        w=float(weights.get(key[0],1.0))
        for idx in range(len(g)-1):
            low=g.iloc[idx]
            high=g.iloc[idx+1]
            delta_b=int(high["bytes"]-low["bytes"])
            benefit=max(
                0.0,
                float(low["max_rank_error"]-high["max_rank_error"])
            )*w
            if delta_b>0:
                upgrades.append({
                    "score":benefit/delta_b,
                    "key":key,
                    "new_idx":idx+1,
                    "delta_b":delta_b,
                })

    upgrades.sort(key=lambda x:x["score"],reverse=True)

    for u in upgrades:
        key=u["key"]
        if selected[key] != u["new_idx"]-1:
            continue
        if used+u["delta_b"] > total_budget:
            continue
        selected[key]=u["new_idx"]
        used+=u["delta_b"]

    rows=[]
    weighted_sum=0.0
    total_weight=0.0
    weighted_hits=0.0

    for key,idx in selected.items():
        name,epoch=key
        row=grouped[key].iloc[idx]
        w=float(weights.get(name,1.0))
        err=float(row["max_rank_error"])
        met=err<=target

        weighted_sum+=w*err
        total_weight+=w
        weighted_hits+=w*(1.0 if met else 0.0)

        rows.append({
            "stream":name,
            "epoch":epoch,
            "weight":w,
            "k":int(row["k"]),
            "bytes":int(row["bytes"]),
            "max_rank_error":err,
            "target_met":met,
        })

    baselines=_uniform_baselines(options,total_budget,target,weights)

    return (
        pd.DataFrame(rows),
        {
            "feasible":True,
            "minimum_required_bytes":min_required,
            "used_bytes":used,
            "weighted_mean_error":weighted_sum/max(total_weight,1e-12),
            "weighted_target_hit_rate":weighted_hits/max(total_weight,1e-12),
        },
        baselines,
    )

def _uniform_baselines(options,total_budget,target,weights):
    rows=[]
    for k in KS:
        g=options[options["k"]==k]
        b=int(g["bytes"].sum())
        weighted_error=0.0
        weighted_hits=0.0
        total_weight=0.0

        for _,r in g.iterrows():
            w=float(weights.get(r["stream"],1.0))
            err=float(r["max_rank_error"])
            weighted_error+=w*err
            weighted_hits+=w*(1.0 if err<=target else 0.0)
            total_weight+=w

        rows.append({
            "method":f"uniform_k_{k}",
            "total_bytes":b,
            "weighted_mean_error":weighted_error/max(total_weight,1e-12),
            "weighted_target_hit_rate":weighted_hits/max(total_weight,1e-12),
            "within_budget":b<=total_budget,
        })

    return pd.DataFrame(rows)

def run_global_budget_sweep(
    streams,
    output_dir,
    budgets=(100000,150000,200000,250000,300000,400000,500000),
    targets=(.01,.005,.0025),
    epoch_sizes=(50000,100000,200000),
    weight_profiles=None,
):
    os.makedirs(output_dir,exist_ok=True)
    weight_profiles=weight_profiles or {
        "uniform":{"trip_total":1,"trip_miles":1,"trip_seconds":1},
        "total_heavy":{"trip_total":4,"trip_miles":1,"trip_seconds":1},
        "seconds_heavy":{"trip_total":1,"trip_miles":1,"trip_seconds":4},
    }

    summary=[]

    for epoch_size in epoch_sizes:
        cache_path=os.path.join(output_dir,f"options_epoch_{epoch_size}.csv")
        options=precompute_options(streams,epoch_size,cache_path)
        min_bytes=minimum_required_bytes(options)

        for target,budget,(profile_name,weights) in itertools.product(
            targets,budgets,weight_profiles.items()
        ):
            allocation,metrics,baselines=allocate_from_options(
                options,int(budget),float(target),weights
            )

            feasible_uniform=baselines[baselines["within_budget"]==True]
            best=(
                feasible_uniform.sort_values("weighted_mean_error").iloc[0]
                if len(feasible_uniform) else None
            )

            best_name=None
            best_err=None
            best_hit=None
            improvement=None
            hit_gain=None

            if metrics["feasible"] and best is not None:
                best_name=str(best["method"])
                best_err=float(best["weighted_mean_error"])
                best_hit=float(best["weighted_target_hit_rate"])
                if best_err>0:
                    improvement=1.0-metrics["weighted_mean_error"]/best_err
                hit_gain=metrics["weighted_target_hit_rate"]-best_hit

            summary.append({
                "objective":OBJECTIVE_NAME,
                "epoch_size":epoch_size,
                "target":target,
                "budget":budget,
                "weight_profile":profile_name,
                "feasible":metrics["feasible"],
                "minimum_required_bytes":metrics["minimum_required_bytes"],
                "used_bytes":metrics["used_bytes"],
                "budget_utilization":(
                    metrics["used_bytes"]/budget
                    if metrics["feasible"] else None
                ),
                "adaptive_weighted_mean_error":metrics["weighted_mean_error"],
                "adaptive_weighted_target_hit_rate":metrics["weighted_target_hit_rate"],
                "best_feasible_uniform":best_name,
                "best_uniform_weighted_mean_error":best_err,
                "best_uniform_weighted_target_hit_rate":best_hit,
                "error_improvement_fraction":improvement,
                "target_hit_rate_gain":hit_gain,
            })

            tag=f"e{epoch_size}_t{target:g}_b{budget}_{profile_name}"
            d=os.path.join(output_dir,"runs",tag)
            os.makedirs(d,exist_ok=True)
            baselines.to_csv(os.path.join(d,"uniform_baselines.csv"),index=False)

            if metrics["feasible"]:
                allocation.to_csv(os.path.join(d,"allocation.csv"),index=False)
            else:
                with open(os.path.join(d,"infeasible.json"),"w") as f:
                    json.dump({
                        "budget":budget,
                        "minimum_required_bytes":metrics["minimum_required_bytes"],
                        "reason":"budget_below_minimum_K64_storage",
                    },f,indent=2)

    sdf=pd.DataFrame(summary)
    sdf.to_csv(os.path.join(output_dir,"global_budget_sweep_summary.csv"),index=False)

    winners=sdf[
        (sdf["feasible"]==True) &
        (sdf["error_improvement_fraction"].fillna(-1)>0) &
        (sdf["target_hit_rate_gain"].fillna(-1)>=0)
    ].copy()

    winners.sort_values(
        ["error_improvement_fraction","target_hit_rate_gain"],
        ascending=False
    ).to_csv(os.path.join(output_dir,"dominant_configs.csv"),index=False)

    with open(os.path.join(output_dir,"sweep_config.json"),"w") as f:
        json.dump({
            "objective":OBJECTIVE_NAME,
            "budgets":list(budgets),
            "targets":list(targets),
            "epoch_sizes":list(epoch_sizes),
            "weight_profiles":weight_profiles,
            "streams":streams,
        },f,indent=2)

    return sdf
