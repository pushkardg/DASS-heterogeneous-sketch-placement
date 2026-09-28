# IEEE BigData 2026 final reproducibility artifact

This branch is the curated artifact for the paper **Workload-Aware Precision Placement for Approximate Analytics over Big-Data Repositories**. It contains only implementation code, public-data experiment code, and measured outputs that support claims in the current manuscript.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

## Commands for the paper-facing results

Run the following commands from the repository root after completing the environment setup above.

### 1. NOAA cross-domain validation

This regenerates the NOAA GHCN-Daily candidate profiles, uniform-tier footprints, and the 45 HASP/LEF configurations used for the cross-domain results.

```bash
bash experiments/noaa_second_dataset/run_noaa_validation.sh
```

Expected outputs:

```text
results/ieee-bigdata-2026/noaa/evaluation/candidate_profiles.csv
results/ieee-bigdata-2026/noaa/evaluation/uniform_tier_footprints.csv
results/ieee-bigdata-2026/noaa/evaluation/per_configuration_results.csv
results/ieee-bigdata-2026/noaa/evaluation/summary.json
results/ieee-bigdata-2026/noaa/evaluation/command_manifest.txt
```

### 2. Fresh-realization robustness check

This is the five-rebuild check used to verify that the heterogeneous-placement advantage is not only an artifact of reusing the KLL realizations that were profiled during allocation.

```bash
PYTHONPATH=src python scripts/run_fresh_realization_check.py \
  --profiles results/chicago-v10/storage-pipeline/2019q1/run/profile/profiles.csv \
  --output-dir results/chicago-v30/fresh-realization \
  --epoch-size 100000 \
  --budget 150000 \
  --target 0.005 \
  --repeats 5
```

Expected outputs:

```text
results/chicago-v30/fresh-realization/fresh_realization_units.csv
results/chicago-v30/fresh-realization/fresh_realization_summary.csv
results/chicago-v30/fresh-realization/summary.json
```

### 3. Repository-scale HASP allocator benchmark

This reproduces the allocator-core scaling experiment from 1K through 1M already-profiled logical units. It intentionally excludes candidate-sketch construction and raw-data profiling.

```bash
PYTHONPATH=src python scripts/run_repository_scale_greedy.py \
  --unit-counts 1000,10000,100000,1000000 \
  --repeats 3 \
  --seed 20260912 \
  --budget-fraction 0.35 \
  --output results/chicago-v13/repository-scale
```

Expected outputs:

```text
results/chicago-v13/repository-scale/repository_scale_runs.csv
results/chicago-v13/repository-scale/repository_scale_summary.csv
results/chicago-v13/repository-scale/metadata.json
```

### 4. HASP+ allocator rerun and NOAA LEF diagnostic

These scripts reuse the committed Chicago and NOAA candidate-profile CSVs. They do not rebuild KLL sketches or perform a controlled seed sweep. The rerun asserts that single-pass HASP reproduces the archived weighted errors within 1e-10, then compares dependency-aware HASP+ to HASP and LEF. The diagnostic decomposes the 21 NOAA configurations where LEF beats HASP+.

```bash
python scripts/run_real_profiles.py
python scripts/noaa_lef_diagnostic.py
```

Expected outputs:

```text
results/ieee-bigdata-2026/allocator-rerun/real_profile_comparison.csv
results/ieee-bigdata-2026/allocator-rerun/real_profile_summary.json
results/ieee-bigdata-2026/allocator-rerun/noaa_lef_diagnostic.csv
results/ieee-bigdata-2026/allocator-rerun/noaa_lef_diagnostic_summary.json
```

### 5. Commands/results retained from the full Chicago evaluation

The manuscript also uses the complete Chicago feasible sweep, allocator comparison, five-period Student-t aggregate, storage/query prototype, HASP-vs-DP scaling, HASP-vs-LEF scaling, matched-budget query benchmark, and the independently profiled 558-unit HASP-vs-LEF quality sweep. Their exact measured CSV/JSON outputs are committed in the result directories listed in the paper-to-artifact map below.

The final-clean branch deliberately contains only the implementation and scripts needed by the current manuscript; it does not claim that a command is rerunnable when the corresponding raw public-data extraction or historical orchestration script was intentionally removed. In particular, do not substitute a synthetic rerun for the committed Chicago measurements. The committed outputs are the evidence for those paper claims.

## Included implementation

`src/dass/` contains only code used by the current paper: KLL construction, tie-aware rank error, HASP/LEF/weight-proportional/exact-DP allocation, global-budget sweeps, temporal-replicate aggregation, independent quality-at-scale evaluation, persistence/query evaluation, matched-budget query comparison, allocator scaling helpers, and the Chicago BigQuery extraction helper.

The NOAA cross-domain experiment is under `experiments/noaa_second_dataset/`.

## Paper-to-artifact map

| Paper evidence | Committed artifact |
|---|---|
| Chicago complete feasible sweep | `results/chicago-v9/full-sweep/` |
| Chicago allocator comparison | `results/chicago-v9/heuristics/` |
| Five-period Student-t aggregate and representative configurations | `results/chicago-v10/aggregate-student-t/` |
| Chicago dataset row/file manifest | `results/chicago-v10/dataset_manifest.csv`, `dataset_manifest_summary.json` |
| File-backed storage prototype and mixed-K round trip | `results/chicago-v10/storage-pipeline/2019q1/run/` |
| 2019Q1 Parquet inputs used by storage/query and fresh-realization checks | `results/chicago-v10/storage-pipeline/2019q1/parquet/` |
| HASP vs quantized-DP runtime scaling | `results/chicago-v10/scalability-q16/` |
| 1K-1M HASP allocator-core scaling | `results/chicago-v13/repository-scale/` |
| HASP vs LEF runtime scaling | `results/chicago-v24/lef-scaling/` |
| Matched-budget query-path benchmark | `results/chicago-v28/matched-query/matched_query_comparison.json` |
| 558-unit independently profiled HASP-vs-LEF quality sweep | `results/chicago-v29/independent-quality-scale/` |
| Fresh-realization robustness check | `results/chicago-v30/fresh-realization/` |
| NOAA provenance | `results/ieee-bigdata-2026/noaa/DATASET_PROVENANCE.txt` |
| NOAA 225-unit candidate profiles, uniform-tier footprints, and 45-config comparison | `results/ieee-bigdata-2026/noaa/evaluation/` |
| HASP+ archived-profile rerun and NOAA LEF diagnostic | `scripts/hasp_plus.py`, `scripts/run_real_profiles.py`, `scripts/noaa_lef_diagnostic.py`, `results/ieee-bigdata-2026/allocator-rerun/` |

The version labels in the Chicago paths identify the exact measured artifact generation that the manuscript cites. Older and unrelated result versions are not included here.

## NOAA reproduction

```bash
bash experiments/noaa_second_dataset/run_noaa_validation.sh
```

The runner contains the requested environment setup. It downloads missing GHCN-Daily yearly inputs, profiles TMAX/PRCP/AWND, derives the three budgets from NOAA's measured uniform-tier footprints, and writes the cross-domain results. Raw NOAA extracts are not committed to this clean branch because they are public and reproducible from NOAA NCEI.

## Chicago data provenance

The Chicago experiment uses the Google BigQuery public table:

`bigquery-public-data.chicago_taxi_trips.taxi_trips`

The evaluated fields are `trip_total`, `trip_miles`, and `trip_seconds`. The five periods and input layout are recorded in `configs/chicago_replicates.json`; extraction logic is in `src/dass/bigquery_source.py`. Public raw Chicago CSV extracts are not duplicated in this branch. The retained 2019Q1 Parquet files are the exact inputs used for the file-backed prototype and the fresh-realization check.

## Standalone robustness and scaling scripts

```bash
python scripts/run_fresh_realization_check.py
python scripts/run_repository_scale_greedy.py
```

The fresh-realization script rebuilds new KLL objects after allocation. The repository-scale benchmark measures only the already-profiled allocation core, not one-million-unit sketch construction.

## Evidence policy

Only values present in committed CSV/JSON outputs are treated as measurements. Missing output is missing evidence. Chicago periods are temporal replications within one domain; NOAA is one additional public domain; the evaluated sketch family is KLL.
