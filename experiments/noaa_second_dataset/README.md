# NOAA second-dataset validation

This directory provides the IEEE BigData paper's cross-domain validation on NOAA GHCN-Daily. It contains no hand-entered paper results: candidate profiles, budgets, and comparisons are computed from the extracted NOAA values.

## Dataset and provenance

Source: NOAA NCEI Global Historical Climatology Network Daily (GHCN-Daily), public bulk by-year archive.

Metrics:
- `TMAX`: daily maximum temperature
- `PRCP`: daily precipitation
- `AWND`: average daily wind speed

Years: 2019--2023. The extractor excludes NOAA observations carrying a quality flag and deterministically caps each metric/year at 2,000,000 values in source order. Source numeric units are retained because rank error is invariant to positive linear unit conversion.

## One-command reproduction

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
bash experiments/noaa_second_dataset/run_noaa_validation.sh
```

The shell script contains the same environment setup at its top, so after cloning the branch the final command alone is sufficient:

```bash
bash experiments/noaa_second_dataset/run_noaa_validation.sh
```

Existing extracted inputs are reused. To redownload all five years:

```bash
FORCE_EXTRACT=1 bash experiments/noaa_second_dataset/run_noaa_validation.sh
```

## Evaluation protocol

The evaluator uses the repository's Apache DataSketches KLL construction, tie-aware rank-error evaluator, HASP allocator, and largest-error-first (LEF) allocator. Candidate precisions are `K={64,128,256,512,1024}` and the default epoch size is 100,000 values.

For every year and metric, every candidate K is independently materialized and measured. The evaluator then computes the measured uniform-K storage footprint. Three validation budgets are derived from NOAA itself as midpoint bytes/unit values between the measured K=64/128, K=128/256, and K=256/512 uniform tiers. Chicago's byte quotas are deliberately not reused.

For each year, each of the three data-derived budgets, and each workload profile (uniform, TMAX-heavy, AWND-heavy), the evaluator compares:

- HASP / marginal benefit per byte
- largest-error-first (LEF)
- best feasible uniform K

All feasible losses as well as wins are retained in the output.

## Measured outputs

The run writes to `results/ieee-bigdata-2026/noaa/evaluation/`:

- `candidate_profiles.csv` — every `(year, metric, epoch, K)` measured byte/error point
- `uniform_tier_footprints.csv` — measured uniform-K footprints by year
- `per_configuration_results.csv` — HASP, LEF, and uniform results for every budget/weight/year configuration
- `summary.json` — aggregate counts and measured mean/median improvements
- `command_manifest.txt` — command/configuration record

Do not update the paper with a NOAA improvement number until these measured files have been generated and inspected. No expected improvement percentage is encoded in the code.

## Paper mapping

The intended Evaluation subsection is **Cross-domain validation**. Its narrow question is whether heterogeneous precision can exploit discrete uniform-tier gaps on a non-taxi numeric dataset. The experiment supports cross-domain evidence for KLL; it does not establish generality across all datasets or sketch families.
