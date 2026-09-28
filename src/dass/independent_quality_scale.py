import json,os,time
import pandas as pd
from .storage_pipeline import profile_partitions
from .allocators import greedy_benefit_per_byte,largest_error_first


def _metric_from_stream(stream):
    for metric in ('trip_total','trip_miles','trip_seconds'):
        if stream.endswith(metric): return metric
    return stream


def _expand_weights(streams,metric_weights):
    return {s:float(metric_weights.get(_metric_from_stream(s),1.0)) for s in streams}


def independent_quality_scale(inputs,output_dir,epoch_size=50000,target=.005,
                              bytes_per_unit=(2000,2500,3000,4000),
                              weight_profiles=None,ks=(64,128,256,512,1024)):
    """Profile every input independently, then compare HASP and LEF quality.

    Input names should encode period and metric, e.g. 2019q1__trip_total.  No
    option table is replicated: every (stream, epoch) unit is built from its
    own data values.  Temporal partitions are distinct repository units, not
    independent statistical domains.
    """
    os.makedirs(output_dir,exist_ok=True)
    if weight_profiles is None:
        weight_profiles={
            'uniform':{'trip_total':1,'trip_miles':1,'trip_seconds':1},
            'total_heavy':{'trip_total':4,'trip_miles':1,'trip_seconds':1},
            'seconds_heavy':{'trip_total':1,'trip_miles':1,'trip_seconds':4},
        }
    t0=time.perf_counter()
    profiles,pmetrics=profile_partitions(inputs,os.path.join(output_dir,'profile'),epoch_size=epoch_size,ks=ks)
    streams=sorted(profiles.stream.unique())
    unit_count=int(profiles[['stream','epoch']].drop_duplicates().shape[0])
    min_required=int(profiles.sort_values('k').groupby(['stream','epoch'],as_index=False).first()['bytes'].sum())
    rows=[]
    for bpu in bytes_per_unit:
        budget=int(bpu)*unit_count
        for profile_name,metric_weights in weight_profiles.items():
            weights=_expand_weights(streams,metric_weights)
            allocations={}
            metrics={}
            for name,fn in [('hasp',greedy_benefit_per_byte),('lef',largest_error_first)]:
                alloc,m=fn(profiles,budget,target,weights); allocations[name]=alloc; metrics[name]=m
                rows.append({'units':unit_count,'epoch_size':epoch_size,'bytes_per_unit':int(bpu),'budget':budget,
                             'weight_profile':profile_name,'method':name,**m})
            if metrics['hasp'].get('within_budget') and metrics['lef'].get('within_budget'):
                he=metrics['hasp']['weighted_mean_error']; le=metrics['lef']['weighted_mean_error']
                winner='hasp' if he<le else ('lef' if le<he else 'tie')
                gap_pct=(le-he)/le*100.0 if le else 0.0
                rows.append({'units':unit_count,'epoch_size':epoch_size,'bytes_per_unit':int(bpu),'budget':budget,
                             'weight_profile':profile_name,'method':'comparison','within_budget':True,
                             'winner':winner,'hasp_error':he,'lef_error':le,
                             'hasp_error_reduction_vs_lef_pct':gap_pct,
                             'hasp_hit_rate':metrics['hasp']['weighted_target_hit_rate'],
                             'lef_hit_rate':metrics['lef']['weighted_target_hit_rate'],
                             'hasp_runtime_ms':metrics['hasp']['runtime_ms'],'lef_runtime_ms':metrics['lef']['runtime_ms']})
    df=pd.DataFrame(rows); df.to_csv(os.path.join(output_dir,'independent_quality_scale.csv'),index=False)
    comp=df[df.method=='comparison'].copy()
    summary={
        'methodology':'independently profiled temporal repository units; no profile-table replication',
        'statistical_scope':'distinct temporal repository units from one Chicago Taxi domain; not independent domains',
        'input_streams':len(inputs),'unit_count':unit_count,'profile_rows':int(len(profiles)),
        'epoch_size':int(epoch_size),'candidate_ks':list(map(int,ks)),'target':float(target),
        'bytes_per_unit':list(map(int,bytes_per_unit)),'minimum_required_bytes':min_required,
        'profiling':pmetrics,'experiment_wall_seconds':time.perf_counter()-t0,
        'comparisons':int(len(comp)),
        'hasp_wins':int((comp.winner=='hasp').sum()) if len(comp) else 0,
        'lef_wins':int((comp.winner=='lef').sum()) if len(comp) else 0,
        'ties':int((comp.winner=='tie').sum()) if len(comp) else 0,
    }
    if len(comp):
        summary['mean_hasp_error_reduction_vs_lef_pct']=float(comp.hasp_error_reduction_vs_lef_pct.mean())
        summary['median_hasp_error_reduction_vs_lef_pct']=float(comp.hasp_error_reduction_vs_lef_pct.median())
        summary['by_budget_and_weight']=comp[['bytes_per_unit','weight_profile','winner','hasp_error','lef_error','hasp_error_reduction_vs_lef_pct','hasp_hit_rate','lef_hit_rate']].to_dict('records')
    with open(os.path.join(output_dir,'independent_quality_scale_summary.json'),'w') as f: json.dump(summary,f,indent=2)
    return df,summary
