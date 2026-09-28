# IEEE BigData 2026 final reproducibility artifact

This branch is the curated artifact for the paper **Workload-Aware Precision Placement for Approximate Analytics over Big-Data Repositories**. It contains only implementation code, public-data experiment code, and measured outputs that support claims in the current manuscript.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

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
