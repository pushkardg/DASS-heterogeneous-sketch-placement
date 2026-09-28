"""Compare LEF and HASP+ selections on archived NOAA candidate profiles."""
import csv
import json
from collections import defaultdict
from pathlib import Path

from hasp_plus import allocate
from run_real_profiles import read_units, NOAA_PROFILES, NOAA_ARCHIVE, OUTPUT

ROOT = Path(__file__).parent
WEIGHTS = {'uniform': {}, 'tmax_heavy': {'tmax': 4}, 'awnd_heavy': {'awnd': 4}}


def lef(units, budget):
    pos = [0] * len(units)
    spent = sum(rows[0][1] for _, rows in units)
    while True:
        candidates = []
        for i, (_, rows) in enumerate(units):
            j = pos[i]
            if j + 1 < len(rows):
                delta = rows[j+1][1] - rows[j][1]
                if spent + delta <= budget:
                    candidates.append((rows[j][3] * rows[j][2], i, delta))
        if not candidates:
            break
        # Original pandas loop traverses groupby keys in sorted order; max keeps first tie.
        _, i, delta = max(candidates, key=lambda x: x[0])
        pos[i] += 1
        spent += delta
    return pos, spent


def main():
    archived = {(int(r['year']), int(r['bytes_per_unit']), r['weight_profile']): r
                for r in csv.DictReader(open(NOAA_ARCHIVE))}
    output = []
    for year in range(2019, 2024):
        for bpu in (2458, 4568, 8730):
            for name, weights in WEIGHTS.items():
                units = read_units(NOAA_PROFILES, year, weights)
                budget = bpu * len(units)
                plus = allocate(units, budget, plus=True)
                pos, lef_bytes = lef(units, budget)
                denom = sum(rows[0][3] for _, rows in units)
                lef_error = sum(rows[j][2]*rows[j][3] for (_, rows), j in zip(units, pos))/denom
                original = float(archived[(year,bpu,name)]['lef_error'])
                if abs(lef_error-original)>1e-10:
                    raise ValueError((year,bpu,name,lef_error,original))
                if lef_error >= plus['weighted_error']-1e-12:
                    continue
                higher = lower = same = nonadjacent = 0
                higher_gain = lower_offset = same_gain = 0.0
                for (unit, rows), j in zip(units, pos):
                    p = next(k for k,r in enumerate(rows) if r[0]==plus['selection'][unit])
                    gain=(rows[p][2]-rows[j][2])*rows[0][3]/denom
                    if j>p:
                        higher+=1;higher_gain+=gain
                    elif j<p:
                        lower+=1;lower_offset+=gain
                    else:
                        same+=1;same_gain+=gain
                    if abs(j-p)>1: nonadjacent+=1
                output.append({'year':year,'bytes_per_unit':bpu,'weight_profile':name,
                               'lef_error':lef_error,'plus_error':plus['weighted_error'],
                               'gap_plus_minus_lef':plus['weighted_error']-lef_error,
                               'lef_bytes':lef_bytes,'plus_bytes':plus['bytes'],
                               'lef_higher_k_units':higher,'lef_lower_k_units':lower,
                               'same_k_units':same,'endpoints_over_one_tier_apart':nonadjacent,
                               'higher_k_gain':higher_gain,'lower_k_offset':lower_offset,
                               'same_k_gain':same_gain})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT/'noaa_lef_diagnostic.csv','w',newline='') as f:
        w=csv.DictWriter(f,output[0].keys());w.writeheader();w.writerows(output)
    summary={'lef_wins':len(output),'budget_counts':dict((b,sum(x['bytes_per_unit']==b for x in output)) for b in (2458,4568,8730)),
             'mean_lef_higher_k_units':sum(x['lef_higher_k_units'] for x in output)/len(output),
             'mean_lef_lower_k_units':sum(x['lef_lower_k_units'] for x in output)/len(output),
             'mean_endpoints_over_one_tier_apart':sum(x['endpoints_over_one_tier_apart'] for x in output)/len(output),
             'total_higher_k_gain':sum(x['higher_k_gain'] for x in output),
             'total_lower_k_offset':sum(x['lower_k_offset'] for x in output),
             'total_gap':sum(x['gap_plus_minus_lef'] for x in output)}
    (OUTPUT/'noaa_lef_diagnostic_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
