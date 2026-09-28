import csv,json,os,time,resource,tracemalloc,platform,sys
from importlib import metadata
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .sketch import build_kll,serialized_bytes
from .evaluator import evaluate_rank_error
from .allocators import greedy_benefit_per_byte

KS=[64,128,256,512,1024]

def _rss_mb():
    # ru_maxrss is KiB on Linux, bytes on macOS; experiments target Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0

def _software_versions():
    pkgs={}
    for name in ('numpy','pandas','pyarrow','datasketches'):
        try:
            pkgs[name]=metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name]=None
    return {
        'python':sys.version.split()[0],
        'platform':platform.platform(),
        'packages':pkgs,
    }

def csv_to_parquet(csv_path,parquet_path,row_group_size=100000):
    os.makedirs(os.path.dirname(parquet_path) or '.',exist_ok=True)
    df=pd.read_csv(csv_path)
    df.to_parquet(parquet_path,index=False,row_group_size=row_group_size)
    return parquet_path

def _iter_values(path,value_column='value',chunk_rows=100000):
    if path.endswith('.parquet'):
        pf=pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=chunk_rows,columns=[value_column]):
            yield np.asarray(batch.column(0).to_numpy(),dtype=float)
    else:
        for df in pd.read_csv(path,usecols=[value_column],chunksize=chunk_rows):
            yield df[value_column].to_numpy(dtype=float)

def profile_partitions(inputs,output_dir,epoch_size=100000,value_column='value',ks=KS):
    os.makedirs(output_dir,exist_ok=True)
    rows=[]; wall0=time.perf_counter(); cpu0=time.process_time(); rss0=_rss_mb(); tracemalloc.start()
    for stream,path in inputs.items():
        carry=np.empty(0,dtype=float); epoch=0
        for chunk in _iter_values(path,value_column,epoch_size):
            x=np.concatenate([carry,chunk])
            start=0
            while len(x)-start>=epoch_size:
                part=x[start:start+epoch_size]; start+=epoch_size
                for k in ks:
                    t0=time.perf_counter(); c0=time.process_time()
                    sk=build_kll(part,k); ev=evaluate_rank_error(sk,part)
                    rows.append({'stream':stream,'epoch':epoch,'k':k,'bytes':serialized_bytes(sk),
                                 'max_rank_error':ev['max_rank_error'],'build_wall_ms':(time.perf_counter()-t0)*1000,
                                 'build_cpu_ms':(time.process_time()-c0)*1000})
                epoch+=1
            carry=x[start:]
        if len(carry)>=100:
            for k in ks:
                t0=time.perf_counter(); c0=time.process_time(); sk=build_kll(carry,k); ev=evaluate_rank_error(sk,carry)
                rows.append({'stream':stream,'epoch':epoch,'k':k,'bytes':serialized_bytes(sk),
                             'max_rank_error':ev['max_rank_error'],'build_wall_ms':(time.perf_counter()-t0)*1000,
                             'build_cpu_ms':(time.process_time()-c0)*1000})
    cur,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    df=pd.DataFrame(rows); df.to_csv(os.path.join(output_dir,'profiles.csv'),index=False)
    summary={'wall_seconds':time.perf_counter()-wall0,'cpu_seconds':time.process_time()-cpu0,
             'peak_rss_mb':_rss_mb(),'rss_delta_mb':max(0.,_rss_mb()-rss0),'tracemalloc_peak_mb':peak/(1024**2),
             'profile_rows':len(df),'candidate_ks':list(ks)}
    with open(os.path.join(output_dir,'profiling_metrics.json'),'w') as f: json.dump(summary,f,indent=2)
    return df,summary

def baseline_materialization_cost(inputs,epoch_size=100000,k=128,value_column='value'):
    wall0=time.perf_counter(); cpu0=time.process_time(); rss0=_rss_mb(); count=0; bytes_out=0
    for stream,path in inputs.items():
        carry=np.empty(0,dtype=float)
        for chunk in _iter_values(path,value_column,epoch_size):
            x=np.concatenate([carry,chunk]); start=0
            while len(x)-start>=epoch_size:
                sk=build_kll(x[start:start+epoch_size],k); bytes_out+=serialized_bytes(sk); count+=1; start+=epoch_size
            carry=x[start:]
        if len(carry)>=100:
            sk=build_kll(carry,k); bytes_out+=serialized_bytes(sk); count+=1
    return {'baseline_k':k,'wall_seconds':time.perf_counter()-wall0,'cpu_seconds':time.process_time()-cpu0,
            'peak_rss_mb':_rss_mb(),'rss_delta_mb':max(0.,_rss_mb()-rss0),'sketches':count,'serialized_bytes':bytes_out}

def materialize_selected(inputs,profiles,output_dir,budget,target,weights,epoch_size=100000,value_column='value'):
    os.makedirs(output_dir,exist_ok=True)
    allocation,metrics=greedy_benefit_per_byte(profiles,budget,target,weights)
    if not metrics.get('within_budget'): raise ValueError('infeasible budget')
    selected={(r.stream,int(r.epoch)):int(r.k) for r in allocation.itertuples()}
    catalog=[]; wall0=time.perf_counter(); cpu0=time.process_time()
    for stream,path in inputs.items():
        carry=np.empty(0,dtype=float); epoch=0
        for chunk in _iter_values(path,value_column,epoch_size):
            x=np.concatenate([carry,chunk]); start=0
            while len(x)-start>=epoch_size:
                part=x[start:start+epoch_size]; start+=epoch_size; k=selected[(stream,epoch)]
                sk=build_kll(part,k); blob=bytes(sk.serialize())
                fname=f'{stream}-epoch-{epoch:05d}-k{k}.kll'; open(os.path.join(output_dir,fname),'wb').write(blob)
                catalog.append({'stream':stream,'epoch':epoch,'k':k,'bytes':len(blob),'file':fname}); epoch+=1
            carry=x[start:]
        if len(carry)>=100 and (stream,epoch) in selected:
            k=selected[(stream,epoch)]; sk=build_kll(carry,k); blob=bytes(sk.serialize()); fname=f'{stream}-epoch-{epoch:05d}-k{k}.kll'; open(os.path.join(output_dir,fname),'wb').write(blob); catalog.append({'stream':stream,'epoch':epoch,'k':k,'bytes':len(blob),'file':fname})
    distinct_ks=sorted({int(x['k']) for x in catalog})
    manifest={'budget':budget,'target':target,'weights':weights,'allocator_metrics':metrics,
              'materialize_wall_seconds':time.perf_counter()-wall0,'materialize_cpu_seconds':time.process_time()-cpu0,
              'total_persisted_bytes':sum(x['bytes'] for x in catalog),'distinct_k_values':distinct_ks,
              'heterogeneous_k_selected':len(distinct_ks)>1,'entries':catalog}
    with open(os.path.join(output_dir,'catalog.json'),'w') as f: json.dump(manifest,f,indent=2)
    return manifest

def query_catalog(catalog_path,quantiles=(0.5,0.9,0.95,0.99),stream=None,repeats=10):
    from datasketches import kll_doubles_sketch
    with open(catalog_path) as f: catalog=json.load(f)
    root=os.path.dirname(catalog_path)
    entries=[e for e in catalog['entries'] if stream is None or e['stream']==stream]
    latencies=[]; result=None; bytes_read=0
    for _ in range(repeats):
        t0=time.perf_counter(); merged=None; bytes_read=0
        for e in entries:
            blob=open(os.path.join(root,e['file']),'rb').read(); bytes_read+=len(blob)
            sk=kll_doubles_sketch.deserialize(blob)
            if merged is None: merged=sk
            else: merged.merge(sk)
        result=[float(merged.get_quantile(float(q))) for q in quantiles] if merged is not None else []
        latencies.append((time.perf_counter()-t0)*1000)
    a=np.asarray(latencies)
    finite=bool(result) and all(np.isfinite(v) for v in result)
    return {'stream':stream,'sketches_read':len(entries),'bytes_read':bytes_read if entries else 0,
            'quantiles':list(quantiles),'values':result,'merge_success':finite,'repeats':repeats,
            'latency_ms_mean':float(a.mean()) if len(a) else None,
            'latency_ms_p50':float(np.quantile(a,.5)) if len(a) else None,'latency_ms_p95':float(np.quantile(a,.95)) if len(a) else None}

def run_pipeline(inputs,output_dir,budget,target,weights,epoch_size=100000,baseline_k=128,value_column='value'):
    os.makedirs(output_dir,exist_ok=True)
    environment=_software_versions()
    with open(os.path.join(output_dir,'environment.json'),'w') as f: json.dump(environment,f,indent=2)
    baseline=baseline_materialization_cost(inputs,epoch_size,baseline_k,value_column)
    profiles,pmetrics=profile_partitions(inputs,os.path.join(output_dir,'profile'),epoch_size,value_column)
    materialized=materialize_selected(inputs,profiles,os.path.join(output_dir,'sketches'),budget,target,weights,epoch_size,value_column)
    query_metrics=[]
    catalog_path=os.path.join(output_dir,'sketches','catalog.json')
    for stream in inputs:
        query_metrics.append(query_catalog(catalog_path,stream=stream))
    with open(os.path.join(output_dir,'query_results.json'),'w') as f: json.dump(query_metrics,f,indent=2)

    entries=materialized.get('entries',[])
    all_files_exist=all(os.path.isfile(os.path.join(output_dir,'sketches',e['file'])) for e in entries)
    bytes_on_disk=sum(os.path.getsize(os.path.join(output_dir,'sketches',e['file'])) for e in entries) if all_files_exist else None
    verification={
        'catalog_entries':len(entries),
        'all_sketch_files_exist':all_files_exist,
        'catalog_bytes':materialized.get('total_persisted_bytes'),
        'bytes_on_disk':bytes_on_disk,
        'catalog_bytes_match_files':all_files_exist and bytes_on_disk==materialized.get('total_persisted_bytes'),
        'within_budget':bool(materialized.get('allocator_metrics',{}).get('within_budget')),
        'distinct_k_values':materialized.get('distinct_k_values',[]),
        'heterogeneous_k_selected':bool(materialized.get('heterogeneous_k_selected')),
        'all_stream_queries_merged':all(q.get('merge_success',False) for q in query_metrics),
    }
    verification['storage_roundtrip_verified']=all([
        verification['catalog_entries']>0,
        verification['all_sketch_files_exist'],
        verification['catalog_bytes_match_files'],
        verification['within_budget'],
        verification['all_stream_queries_merged'],
    ])
    verification['heterogeneous_storage_claim_verified']=verification['storage_roundtrip_verified'] and verification['heterogeneous_k_selected']
    with open(os.path.join(output_dir,'verification.json'),'w') as f: json.dump(verification,f,indent=2)

    result={'environment':environment,'baseline':baseline,'profiling':pmetrics,
            'selected':{k:v for k,v in materialized.items() if k!='entries'},
            'queries':query_metrics,'verification':verification}
    result['profiling_wall_overhead_x']=pmetrics['wall_seconds']/max(baseline['wall_seconds'],1e-12)
    result['profiling_cpu_overhead_x']=pmetrics['cpu_seconds']/max(baseline['cpu_seconds'],1e-12)
    with open(os.path.join(output_dir,'pipeline_summary.json'),'w') as f: json.dump(result,f,indent=2)
    return result
