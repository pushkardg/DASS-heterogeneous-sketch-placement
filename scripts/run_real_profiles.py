"""Re-run HASP and HASP+ against archived Chicago/NOAA candidate profiles."""
import csv
import json
from collections import defaultdict
from pathlib import Path

from hasp_plus import allocate


ROOT = Path(__file__).parent
IN_REPO = not (ROOT / 'noaa_profiles.csv').exists()
NOAA_PROFILES = (ROOT.parent / 'results/ieee-bigdata-2026/noaa/evaluation/candidate_profiles.csv'
                 if IN_REPO else ROOT / 'noaa_profiles.csv')
CHICAGO_PROFILES = (ROOT.parent / 'results/chicago-v29/independent-quality-scale/profile/profiles.csv'
                    if IN_REPO else ROOT / 'chicago_profiles.csv')
NOAA_ARCHIVE = (ROOT.parent / 'results/ieee-bigdata-2026/noaa/evaluation/per_configuration_results.csv'
                if IN_REPO else ROOT / 'per_configuration_results.csv')
CHICAGO_ARCHIVE = (ROOT.parent / 'results/chicago-v29/independent-quality-scale/independent_quality_scale.csv'
                   if IN_REPO else ROOT / 'independent_quality_scale.csv')
OUTPUT = ROOT.parent / 'results/ieee-bigdata-2026/allocator-rerun' if IN_REPO else ROOT


def read_units(path, year=None, weights=None):
    groups = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if year is not None and not r['stream'].startswith(str(year) + '__'):
                continue
            metric = r['stream'].split('__')[-1]
            w = weights.get(metric, 1.0)
            groups[(r['stream'], int(r['epoch']))].append(
                (int(r['k']), int(r['bytes']), float(r['max_rank_error']), float(w)))
    return [(f'{stream}:{epoch}', sorted(rows)) for (stream, epoch), rows in sorted(groups.items())]


def main():
    chicago = {
        'uniform': {}, 'total_heavy': {'trip_total': 4},
        'seconds_heavy': {'trip_seconds': 4},
    }
    noaa = {
        'uniform': {}, 'tmax_heavy': {'tmax': 4},
        'awnd_heavy': {'awnd': 4},
    }
    archived = {}
    for r in csv.DictReader(open(NOAA_ARCHIVE)):
        archived[(int(r['year']), int(r['bytes_per_unit']), r['weight_profile'])] = r
    archived_chicago = {}
    for r in csv.DictReader(open(CHICAGO_ARCHIVE)):
        if r['method'] == 'comparison':
            archived_chicago[(int(r['bytes_per_unit']), r['weight_profile'])] = r
    result = []
    for domain, years, budgets, weight_sets, filename in (
        ('noaa', range(2019, 2024), [2458, 4568, 8730], noaa, NOAA_PROFILES),
        ('chicago', [None], [2000, 2500, 3000, 4000], chicago, CHICAGO_PROFILES),
    ):
        for year in years:
            for name, weights in weight_sets.items():
                units = read_units(filename, year, weights)
                for bpu in budgets:
                    budget = bpu * len(units)
                    old = allocate(units, budget)
                    plus = allocate(units, budget, plus=True)
                    original = (archived[(year, bpu, name)]['hasp_error'] if domain == 'noaa'
                                else archived_chicago[(bpu, name)]['hasp_error'])
                    difference = abs(old['weighted_error'] - float(original))
                    if difference > 1e-10:
                        raise ValueError(f'Baseline reproduction failed: {domain} {year} {bpu} {name}: {difference}')
                    lef = float(archived[(year, bpu, name)]['lef_error'] if domain == 'noaa'
                                else archived_chicago[(bpu, name)]['lef_error'])
                    result.append({'domain': domain, 'year': year, 'bytes_per_unit': bpu,
                                   'weight_profile': name, 'units': len(units), 'budget': budget,
                                   'hasp_error': old['weighted_error'], 'hasp_plus_error': plus['weighted_error'],
                                   'lef_error': lef, 'hasp_bytes': old['bytes'], 'hasp_plus_bytes': plus['bytes'],
                                   'predecessor_skips': old['skipped_predecessor'],
                                   'plus_gain_vs_hasp_pct': 100*(old['weighted_error']-plus['weighted_error'])/old['weighted_error'],
                                   'plus_gain_vs_lef_pct': 100*(lef-plus['weighted_error'])/lef})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT / 'real_profile_comparison.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result[0].keys()); writer.writeheader();writer.writerows(result)
    summary = {}
    for domain in ('noaa', 'chicago'):
        rows = [r for r in result if r['domain']==domain]
        summary[domain] = {
            'configurations': len(rows),
            'hasp_plus_wins_vs_hasp': sum(r['hasp_plus_error'] < r['hasp_error']-1e-12 for r in rows),
            'hasp_wins_vs_hasp_plus': sum(r['hasp_error'] < r['hasp_plus_error']-1e-12 for r in rows),
            'hasp_plus_wins_vs_lef': sum(r['hasp_plus_error'] < r['lef_error']-1e-12 for r in rows),
            'lef_wins_vs_hasp_plus': sum(r['lef_error'] < r['hasp_plus_error']-1e-12 for r in rows),
            'mean_plus_gain_vs_hasp_pct': sum(r['plus_gain_vs_hasp_pct'] for r in rows)/len(rows),
            'mean_plus_gain_vs_lef_pct': sum(r['plus_gain_vs_lef_pct'] for r in rows)/len(rows),
            'total_predecessor_skips': sum(r['predecessor_skips'] for r in rows),
        }
    (OUTPUT / 'real_profile_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
