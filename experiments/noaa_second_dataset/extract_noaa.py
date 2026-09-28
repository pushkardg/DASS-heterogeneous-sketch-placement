#!/usr/bin/env python3
"""Extract reproducible NOAA daily climate streams for HASP cross-domain validation.

Input is NOAA's public GHCN-Daily by-year CSV archive. The script keeps three
measurements with different distributions: TMAX (daily max temperature), PRCP
(precipitation), and AWND (average wind). It writes one float per line so the
existing KLL profiling pipeline can consume the files without dataset-specific
changes.

No paper result is encoded here: this script only prepares input data.
"""
import argparse
import csv
import gzip
import io
import pathlib
import urllib.error
import urllib.request

ELEMENTS = {"TMAX": "tmax", "PRCP": "prcp", "AWND": "awnd"}
# Official NCEI GHCN-Daily bulk-by-year directory.
URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/{year}.csv.gz"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--max-values-per-metric", type=int, default=2_000_000,
                   help="Deterministic cap in source order; 0 means no cap")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    url = URL.format(year=args.year)
    print(f"Downloading {url}")
    outs = {e: open(args.output / f"noaa_{args.year}_{name}.txt", "w") for e, name in ELEMENTS.items()}
    counts = {e: 0 for e in ELEMENTS}
    try:
        try:
            response = urllib.request.urlopen(url, timeout=120)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"NOAA download failed ({exc.code}) for {url}") from exc
        with response:
            with gzip.GzipFile(fileobj=response) as gz:
                reader = csv.reader(io.TextIOWrapper(gz, encoding="utf-8"))
                # GHCN by-year format has no header:
                # ID,DATE,ELEMENT,DATA_VALUE,M_FLAG,Q_FLAG,S_FLAG,OBS_TIME
                for row in reader:
                    if len(row) < 4:
                        continue
                    e = row[2]
                    if e not in ELEMENTS:
                        continue
                    if args.max_values_per_metric and counts[e] >= args.max_values_per_metric:
                        continue
                    # Exclude values carrying a NOAA quality flag.
                    if len(row) > 5 and row[5].strip():
                        continue
                    try:
                        v = float(row[3])
                    except (ValueError, TypeError):
                        continue
                    # Retain source numeric units. Rank error is invariant to
                    # positive linear unit conversion.
                    outs[e].write(f"{v}\n")
                    counts[e] += 1
                    if args.max_values_per_metric and all(
                        counts[x] >= args.max_values_per_metric for x in ELEMENTS
                    ):
                        break
    finally:
        for f in outs.values():
            f.close()

    print("year,metric,count")
    for e, name in ELEMENTS.items():
        print(f"{args.year},{name},{counts[e]}")
    missing = [ELEMENTS[e] for e in ELEMENTS if counts[e] == 0]
    if missing:
        raise RuntimeError(f"No values extracted for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
