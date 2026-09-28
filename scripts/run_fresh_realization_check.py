#!/usr/bin/env python3
"""Fresh-realization robustness check for the 2019Q1 storage-pipeline profile.

The profiling table is used only to choose the HASP assignment.  The selected
sketches are then rebuilt from raw Parquet values, producing fresh randomized
KLL realizations, and their errors are measured again.  A fresh best-feasible
uniform-K baseline is rebuilt from the same data for comparison.

Run from repository root:
  PYTHONPATH=src python scripts/run_fresh_realization_check.py
"""
import argparse, json, os
import numpy as np
import pandas as pd
from dass.allocators import greedy_benefit_per_byte
from dass.evaluator import evaluate_rank_error
from dass.sketch import build_kll, serialized_bytes
from dass.storage_pipeline import _iter_values

KS=(64,128,256,512,1024)
DEFAULT_INPUTS={
    'trip_total':'results/chicago-v10/storage-pipeline/2019q1/parquet/trip_total.parquet',
    'trip_miles':'results/chicago-v10/storage-pipeline/2019q1/parquet/trip_miles.parquet',
    'trip_seconds':'results/chicago-v10/storage-pipeline/2019q1/parquet/trip_seconds.parquet',
}

def partitions(path, epoch_size):
    carry=np.empty(0,dtype=float); epoch=0
    for chunk in _iter_values(path,'value',epoch_size):
        x=np.concatenate([carry,chunk]); start=0
        while len(x)-start>=epoch_size:
            yield epoch,x[start:start+epoch_size]
            epoch+=1; start+=epoch_size
        carry=x[start:]
    if len(carry)>=100:
        yield epoch,carry

def weighted_error(rows,weights):
    num=den=0.0
    for r in rows:
        w=float(weights.get(r['stream'],1.0)); num+=w*r['max_rank_error']; den+=w
    return num/den

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--profiles',default='results/chicago-v10/storage-pipeline/2019q1/run/profile/profiles.csv')
    ap.add_argument('--output-dir',default='results/chicago-v30/fresh-realization')
    ap.add_argument('--epoch-size',type=int,default=100000)
    ap.add_argument('--budget',type=int,default=150000)
    ap.add_argument('--target',type=float,default=0.005)
    ap.add_argument('--repeats',type=int,default=5,help='Independent fresh rebuilds')
    args=ap.parse_args()
    os.makedirs(args.output_dir,exist_ok=True)
    weights={'trip_total':4.0,'trip_miles':1.0,'trip_seconds':1.0}
    profiles=pd.read_csv(args.profiles)
    allocation,am=greedy_benefit_per_byte(profiles,args.budget,args.target,weights)
    if not am.get('within_budget'): raise SystemExit('HASP assignment infeasible')
    selected={(r.stream,int(r.epoch)):int(r.k) for r in allocation.itertuples()}

    # Best uniform tier according to the same measured profile table and budget.
    uniform_candidates=[]
    for k in KS:
        g=profiles[profiles.k==k]
        if len(g)!=len(selected): continue
        b=int(g.bytes.sum())
        if b<=args.budget:
            uniform_candidates.append((k,b))
    if not uniform_candidates: raise SystemExit('No feasible uniform K')
    uniform_k,profile_uniform_bytes=max(uniform_candidates,key=lambda x:x[0])

    profile_hasp_error=float(am['weighted_mean_error'])
    profile_uniform=profiles[profiles.k==uniform_k]
    profile_uniform_rows=[{'stream':r.stream,'max_rank_error':float(r.max_rank_error)} for r in profile_uniform.itertuples()]
    profile_uniform_error=weighted_error(profile_uniform_rows,weights)
    profile_gain=(profile_uniform_error-profile_hasp_error)/profile_uniform_error

    all_rows=[]; summaries=[]
    for rep in range(args.repeats):
        hasp_rows=[]; uniform_rows=[]; hasp_bytes=uniform_bytes=0
        for stream,path in DEFAULT_INPUTS.items():
            for epoch,x in partitions(path,args.epoch_size):
                k=selected[(stream,epoch)]
                hs=build_kll(x,k); he=evaluate_rank_error(hs,x)['max_rank_error']; hb=serialized_bytes(hs)
                us=build_kll(x,uniform_k); ue=evaluate_rank_error(us,x)['max_rank_error']; ub=serialized_bytes(us)
                hasp_rows.append({'stream':stream,'epoch':epoch,'max_rank_error':float(he)})
                uniform_rows.append({'stream':stream,'epoch':epoch,'max_rank_error':float(ue)})
                hasp_bytes+=hb; uniform_bytes+=ub
                all_rows += [
                    {'repeat':rep,'method':'HASP-fresh','stream':stream,'epoch':epoch,'k':k,'bytes':hb,'max_rank_error':he},
                    {'repeat':rep,'method':f'uniform-K{uniform_k}-fresh','stream':stream,'epoch':epoch,'k':uniform_k,'bytes':ub,'max_rank_error':ue},
                ]
        he=weighted_error(hasp_rows,weights); ue=weighted_error(uniform_rows,weights)
        gain=(ue-he)/ue
        summaries.append({'repeat':rep,'fresh_hasp_weighted_error':he,'fresh_uniform_weighted_error':ue,
                          'fresh_error_improvement_fraction':gain,'fresh_hasp_bytes':hasp_bytes,
                          'fresh_uniform_bytes':uniform_bytes,'fresh_hasp_within_original_budget':hasp_bytes<=args.budget})

    pd.DataFrame(all_rows).to_csv(os.path.join(args.output_dir,'fresh_realization_units.csv'),index=False)
    sdf=pd.DataFrame(summaries); sdf.to_csv(os.path.join(args.output_dir,'fresh_realization_summary.csv'),index=False)
    out={
      'budget':args.budget,'target':args.target,'epoch_size':args.epoch_size,'weights':weights,
      'repeats':args.repeats,'uniform_k':uniform_k,'profile_uniform_bytes':profile_uniform_bytes,
      'profile_hasp_weighted_error':profile_hasp_error,'profile_uniform_weighted_error':profile_uniform_error,
      'profile_error_improvement_fraction':profile_gain,
      'fresh_hasp_weighted_error_mean':float(sdf.fresh_hasp_weighted_error.mean()),
      'fresh_uniform_weighted_error_mean':float(sdf.fresh_uniform_weighted_error.mean()),
      'fresh_error_improvement_fraction_mean':float(sdf.fresh_error_improvement_fraction.mean()),
      'fresh_error_improvement_fraction_min':float(sdf.fresh_error_improvement_fraction.min()),
      'fresh_error_improvement_fraction_max':float(sdf.fresh_error_improvement_fraction.max()),
      'gain_retained_fraction':float(sdf.fresh_error_improvement_fraction.mean()/profile_gain) if profile_gain else None,
      'note':'Fresh KLL objects are rebuilt after allocation; profiling realizations are not reused for evaluation.'
    }
    with open(os.path.join(args.output_dir,'summary.json'),'w') as f: json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
