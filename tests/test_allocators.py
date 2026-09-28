import pandas as pd
from dass.allocators import greedy_benefit_per_byte,exact_dp,largest_error_first,weight_proportional


def sample_options():
    rows=[]
    for stream,errs in {'a':[.10,.04,.03],'b':[.08,.07,.02]}.items():
        for k,b,e in zip([64,128,256],[100,180,300],errs):
            rows.append({'stream':stream,'epoch':0,'k':k,'bytes':b,'max_rank_error':e})
    return pd.DataFrame(rows)


def test_infeasible_budget_is_rejected():
    _,m=greedy_benefit_per_byte(sample_options(),199,.05,{'a':1,'b':1})
    assert m['within_budget'] is False
    assert m['minimum_required_bytes']==200


def test_exact_no_worse_than_greedy():
    options=sample_options()
    _,g=greedy_benefit_per_byte(options,480,.05,{'a':1,'b':1})
    _,o=exact_dp(options,480,.05,{'a':1,'b':1},granularity=1)
    assert o['within_budget'] and g['within_budget']
    assert o['weighted_mean_error'] <= g['weighted_mean_error'] + 1e-12


def test_all_cheap_allocators_respect_budget():
    options=sample_options()
    for fn in (greedy_benefit_per_byte,largest_error_first,weight_proportional):
        _,m=fn(options,480,.05,{'a':2,'b':1})
        assert m['within_budget']
        assert m['used_bytes']<=480
