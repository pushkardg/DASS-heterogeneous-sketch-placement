import math,time
import pandas as pd

KS=[64,128,256,512,1024]

def _groups(options):
    return {(s,int(e)):g.sort_values('k').reset_index(drop=True)
            for (s,e),g in options.groupby(['stream','epoch'])}

def _finish(groups,selected,budget,target,weights,method,elapsed):
    rows=[]; we=0.; wh=0.; tw=0.; used=0
    for key,idx in selected.items():
        s,e=key; r=groups[key].iloc[idx]; w=float(weights.get(s,1.0)); err=float(r.max_rank_error)
        used+=int(r.bytes); we+=w*err; wh+=w*(err<=target); tw+=w
        rows.append({'stream':s,'epoch':e,'k':int(r.k),'bytes':int(r.bytes),'max_rank_error':err,'weight':w})
    return pd.DataFrame(rows),{'method':method,'used_bytes':used,'weighted_mean_error':we/max(tw,1e-12),'weighted_target_hit_rate':wh/max(tw,1e-12),'runtime_ms':elapsed*1000,'within_budget':used<=budget}

def greedy_benefit_per_byte(options,budget,target,weights):
    t=time.perf_counter(); groups=_groups(options); selected={k:0 for k in groups}
    used=sum(int(g.iloc[0].bytes) for g in groups.values())
    if used>budget: return pd.DataFrame(),{'method':'benefit_per_byte','within_budget':False,'minimum_required_bytes':used,'runtime_ms':(time.perf_counter()-t)*1000}
    upgrades=[]
    for key,g in groups.items():
        w=float(weights.get(key[0],1.0))
        for j in range(len(g)-1):
            a,b=g.iloc[j],g.iloc[j+1]; db=int(b.bytes-a.bytes); benefit=w*max(0.,float(a.max_rank_error-b.max_rank_error))
            if db>0: upgrades.append((benefit/db,key,j+1,db))
    upgrades.sort(reverse=True,key=lambda x:x[0])
    for _,key,new_idx,db in upgrades:
        if selected[key]==new_idx-1 and used+db<=budget:
            selected[key]=new_idx; used+=db
    return _finish(groups,selected,budget,target,weights,'benefit_per_byte',time.perf_counter()-t)

def largest_error_first(options,budget,target,weights):
    t=time.perf_counter(); groups=_groups(options); selected={k:0 for k in groups}
    used=sum(int(g.iloc[0].bytes) for g in groups.values())
    if used>budget: return pd.DataFrame(),{'method':'largest_error_first','within_budget':False,'minimum_required_bytes':used,'runtime_ms':(time.perf_counter()-t)*1000}
    while True:
        candidates=[]
        for key,idx in selected.items():
            g=groups[key]
            if idx+1<len(g):
                a,b=g.iloc[idx],g.iloc[idx+1]; db=int(b.bytes-a.bytes)
                score=float(weights.get(key[0],1.0))*float(a.max_rank_error)
                if used+db<=budget: candidates.append((score,key,db))
        if not candidates: break
        _,key,db=max(candidates,key=lambda x:x[0]); selected[key]+=1; used+=db
    return _finish(groups,selected,budget,target,weights,'largest_error_first',time.perf_counter()-t)

def weight_proportional(options,budget,target,weights):
    t=time.perf_counter(); groups=_groups(options); selected={k:0 for k in groups}
    used=sum(int(g.iloc[0].bytes) for g in groups.values())
    if used>budget: return pd.DataFrame(),{'method':'weight_proportional','within_budget':False,'minimum_required_bytes':used,'runtime_ms':(time.perf_counter()-t)*1000}
    remaining=budget-used
    total=sum(float(weights.get(k[0],1.0)) for k in groups)
    for key,g in sorted(groups.items(),key=lambda kv:float(weights.get(kv[0][0],1.0)),reverse=True):
        share=remaining*float(weights.get(key[0],1.0))/max(total,1e-12)
        base=int(g.iloc[0].bytes); best=0
        for j in range(1,len(g)):
            if int(g.iloc[j].bytes)-base<=share and used+(int(g.iloc[j].bytes)-base)<=budget: best=j
        used+=int(g.iloc[best].bytes)-base; selected[key]=best
    return _finish(groups,selected,budget,target,weights,'weight_proportional',time.perf_counter()-t)

def exact_dp(options,budget,target,weights,granularity=4):
    t=time.perf_counter(); groups=_groups(options); items=list(groups.items())
    min_bytes=sum(int(g.iloc[0].bytes) for _,g in items)
    if min_bytes>budget: return pd.DataFrame(),{'method':'exact_dp','within_budget':False,'minimum_required_bytes':min_bytes,'runtime_ms':(time.perf_counter()-t)*1000}
    cap=budget//granularity
    dp={0:(0.0,[])}
    for key,g in items:
        nd={}
        w=float(weights.get(key[0],1.0))
        for spent,(err,path) in dp.items():
            for j,r in g.iterrows():
                c=int(math.ceil(int(r.bytes)/granularity)); ns=spent+c
                if ns>cap: continue
                ne=err+w*float(r.max_rank_error)
                if ns not in nd or ne<nd[ns][0]: nd[ns]=(ne,path+[j])
        # Pareto prune by ascending cost / decreasing error
        best=float('inf'); pruned={}
        for c in sorted(nd):
            if nd[c][0]<best: pruned[c]=nd[c]; best=nd[c][0]
        dp=pruned
    spent,(err,path)=min(dp.items(),key=lambda kv:kv[1][0])
    selected={items[i][0]:int(path[i]) for i in range(len(items))}
    return _finish(groups,selected,budget,target,weights,'exact_dp',time.perf_counter()-t)

def compare_allocators(options,budget,target,weights,dp_granularity=4):
    funcs=[greedy_benefit_per_byte,largest_error_first,weight_proportional]
    results=[]
    for f in funcs:
        _,m=f(options,budget,target,weights); results.append(m)
    _,m=exact_dp(options,budget,target,weights,dp_granularity); results.append(m)
    return pd.DataFrame(results)
