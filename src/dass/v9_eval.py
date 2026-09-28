import json,os,time
import numpy as np
import pandas as pd
from .allocators import compare_allocators,greedy_benefit_per_byte,largest_error_first,exact_dp


def full_sweep_report(replicate_csv,output_dir):
    os.makedirs(output_dir,exist_ok=True)
    df=pd.read_csv(replicate_csv)
    f=df[df['feasible']==True].copy()
    f['adaptive_wins_error']=f['error_improvement_fraction']>0
    f['adaptive_wins_both']=(f['error_improvement_fraction']>0)&(f['target_hit_rate_gain']>=0)
    overall={
        'feasible_runs':int(len(f)),
        'error_win_fraction':float(f['adaptive_wins_error'].mean()) if len(f) else None,
        'joint_win_fraction':float(f['adaptive_wins_both'].mean()) if len(f) else None,
        'mean_error_improvement_fraction':float(f['error_improvement_fraction'].mean()) if len(f) else None,
        'median_error_improvement_fraction':float(f['error_improvement_fraction'].median()) if len(f) else None,
        'p10_error_improvement_fraction':float(f['error_improvement_fraction'].quantile(.10)) if len(f) else None,
        'p90_error_improvement_fraction':float(f['error_improvement_fraction'].quantile(.90)) if len(f) else None,
        'losing_runs':int((~f['adaptive_wins_error']).sum()),
    }
    by_epoch=f.groupby('epoch_size').agg(
        runs=('error_improvement_fraction','size'),
        win_fraction=('adaptive_wins_error','mean'),
        mean_improvement=('error_improvement_fraction','mean'),
        median_improvement=('error_improvement_fraction','median'),
        min_improvement=('error_improvement_fraction','min'),
        max_improvement=('error_improvement_fraction','max')).reset_index()
    by_epoch.to_csv(os.path.join(output_dir,'full_sweep_by_epoch.csv'),index=False)
    f.to_csv(os.path.join(output_dir,'full_feasible_grid.csv'),index=False)
    with open(os.path.join(output_dir,'full_sweep_summary.json'),'w') as h: json.dump(overall,h,indent=2)
    return overall

def heuristic_sweep(options_csv,output_dir,budgets,targets,weight_profiles,dp_granularity=4):
    os.makedirs(output_dir,exist_ok=True); options=pd.read_csv(options_csv); rows=[]
    for b in budgets:
        for t in targets:
            for name,w in weight_profiles.items():
                r=compare_allocators(options,int(b),float(t),w,dp_granularity)
                r['budget']=b; r['target']=t; r['weight_profile']=name
                r['dp_granularity_bytes']=int(dp_granularity)
                rows.append(r)
    out=pd.concat(rows,ignore_index=True); out.to_csv(os.path.join(output_dir,'allocator_comparison.csv'),index=False)
    return out

def scalability_benchmark(options_csv,output_dir,unit_counts=(30,60,120,240),budgets_per_unit=(2500,),repeats=3,dp_granularity=16,include_dp=True):
    """Benchmark HASP and LEF on exactly the same expanded option tables and budgets.

    DP is retained as an optional reference.  Each row records the actual allocator
    runtime returned by the allocator, so LEF/HASP comparisons exclude CSV loading
    and synthetic table expansion.  This is allocator-core scalability, not an
    end-to-end repository benchmark.
    """
    os.makedirs(output_dir,exist_ok=True); src=pd.read_csv(options_csv); keys=list(src.groupby(['stream','epoch']).groups.keys()); rows=[]
    if not keys: raise ValueError('no units in options file')
    dp_method='exact_dp' if int(dp_granularity)==1 else f'quantized_dp_{int(dp_granularity)}B'
    for n in unit_counts:
        reps=int(np.ceil(n/len(keys)))
        expanded=[]
        for r in range(reps):
            x=src.copy(); x['epoch']=x['epoch'].astype(int)+r*100000; expanded.append(x)
        data=pd.concat(expanded,ignore_index=True)
        chosen=list(data.groupby(['stream','epoch']).groups.keys())[:n]
        chosen_set=set(chosen); data=data[data.apply(lambda z:(z['stream'],int(z['epoch'])) in chosen_set,axis=1)]
        weights={s:1.0 for s in data['stream'].unique()}
        min_cost=int(data[data['k']==data['k'].min()]['bytes'].sum())
        for bpu in budgets_per_unit:
            budget=max(min_cost,int(n*bpu))
            for rep in range(repeats):
                _,gm=greedy_benefit_per_byte(data,budget,.005,weights)
                rows.append({'units':n,'budget':budget,'repeat':rep,'method':'hasp','dp_granularity_bytes':int(dp_granularity),'runtime_ms':gm.get('runtime_ms'),'weighted_mean_error':gm.get('weighted_mean_error'),'within_budget':gm.get('within_budget')})
                _,lm=largest_error_first(data,budget,.005,weights)
                rows.append({'units':n,'budget':budget,'repeat':rep,'method':'largest_error_first','dp_granularity_bytes':int(dp_granularity),'runtime_ms':lm.get('runtime_ms'),'weighted_mean_error':lm.get('weighted_mean_error'),'within_budget':lm.get('within_budget')})
                if include_dp:
                    try:
                        _,dm=exact_dp(data,budget,.005,weights,dp_granularity)
                        rows.append({'units':n,'budget':budget,'repeat':rep,'method':dp_method,'dp_granularity_bytes':int(dp_granularity),'runtime_ms':dm.get('runtime_ms'),'weighted_mean_error':dm.get('weighted_mean_error'),'within_budget':dm.get('within_budget')})
                    except Exception as e:
                        rows.append({'units':n,'budget':budget,'repeat':rep,'method':dp_method,'dp_granularity_bytes':int(dp_granularity),'runtime_ms':None,'weighted_mean_error':None,'within_budget':None,'error':str(e)})
    out=pd.DataFrame(rows); out.to_csv(os.path.join(output_dir,'allocator_scalability.csv'),index=False)
    summary=out.groupby(['units','method','dp_granularity_bytes']).agg(
        runtime_ms_mean=('runtime_ms','mean'),
        runtime_ms_p50=('runtime_ms',lambda x:x.quantile(.50)),
        runtime_ms_p95=('runtime_ms',lambda x:x.quantile(.95)),
        error_mean=('weighted_mean_error','mean')).reset_index()
    summary.to_csv(os.path.join(output_dir,'allocator_scalability_summary.csv'),index=False)
    hasp=summary[summary.method=='hasp'][['units','runtime_ms_mean']].rename(columns={'runtime_ms_mean':'hasp_runtime_ms_mean'})
    lef=summary[summary.method=='largest_error_first'][['units','runtime_ms_mean']].rename(columns={'runtime_ms_mean':'lef_runtime_ms_mean'})
    comparison=hasp.merge(lef,on='units',how='inner')
    comparison['lef_over_hasp_runtime_x']=comparison['lef_runtime_ms_mean']/comparison['hasp_runtime_ms_mean']
    comparison.to_csv(os.path.join(output_dir,'lef_vs_hasp_scaling.csv'),index=False)
    metadata={
        'options_csv':options_csv,
        'unit_counts':[int(x) for x in unit_counts],
        'bytes_per_unit':[int(x) for x in budgets_per_unit],
        'repeats':int(repeats),
        'dp_granularity_bytes':int(dp_granularity),
        'dp_method_label':dp_method if include_dp else None,
        'include_dp':bool(include_dp),
        'methods':['hasp','largest_error_first']+([dp_method] if include_dp else []),
        'note':'HASP and LEF use identical option tables and budgets. Timings are allocator-core only. granularity=1 is byte-exact DP; granularity>1 is conservative quantized DP used for runtime scaling.'
    }
    with open(os.path.join(output_dir,'scalability_metadata.json'),'w') as f: json.dump(metadata,f,indent=2)
    return out
