#!/usr/bin/env bash
set -euo pipefail

# Repository environment setup requested for reproducibility.
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src

YEARS="${YEARS:-2019 2020 2021 2022 2023}"
EPOCH_SIZE="${EPOCH_SIZE:-100000}"
TARGET="${TARGET:-0.005}"
OUT="${OUT:-results/ieee-bigdata-2026/noaa}"
DATA="$OUT/data"
EVAL="$OUT/evaluation"
mkdir -p "$DATA" "$EVAL"

# Download/extract only missing inputs, so an already-generated dataset is not
# fetched again. Set FORCE_EXTRACT=1 to rebuild every yearly input.
for y in $YEARS; do
  if [[ "${FORCE_EXTRACT:-0}" == "1" || ! -s "$DATA/$y/noaa_${y}_tmax.txt" || ! -s "$DATA/$y/noaa_${y}_prcp.txt" || ! -s "$DATA/$y/noaa_${y}_awnd.txt" ]]; then
    python experiments/noaa_second_dataset/extract_noaa.py --year "$y" --output "$DATA/$y"
  else
    echo "Using existing NOAA inputs for $y"
  fi
done

cat > "$OUT/DATASET_PROVENANCE.txt" <<EOF
Dataset: NOAA Global Historical Climatology Network Daily (GHCN-Daily)
Source: NOAA NCEI GHCN-Daily bulk by-year CSV archive
Years: $YEARS
Metrics: TMAX, PRCP, AWND
Epoch size: $EPOCH_SIZE values
Candidate K values: 64,128,256,512,1024
Purpose: cross-domain validation of the precision-gap / heterogeneous-placement result.
Budget rule: derived from measured NOAA uniform-tier footprints; Chicago quotas are not reused.
EOF

python experiments/noaa_second_dataset/evaluate_noaa.py \
  --data-root "$DATA" \
  --output "$EVAL" \
  --years $YEARS \
  --epoch-size "$EPOCH_SIZE" \
  --target "$TARGET"

echo
echo "NOAA cross-domain evaluation complete."
echo "Primary outputs:"
echo "  $EVAL/candidate_profiles.csv"
echo "  $EVAL/uniform_tier_footprints.csv"
echo "  $EVAL/per_configuration_results.csv"
echo "  $EVAL/summary.json"
echo "  $EVAL/command_manifest.txt"
