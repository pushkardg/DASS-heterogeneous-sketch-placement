"""Compare single-pass HASP with dependency-aware HASP+ on measured profiles.

CSV columns: unit,k,bytes,error,weight (weight optional, defaults to 1).
All costs must strictly increase with k. Error may be nonmonotone.
Usage: python hasp_plus.py profiles.csv --budget 150000 --output results.json
"""
import argparse
import csv
import heapq
import json
from collections import defaultdict


def load_profiles(path):
    units = defaultdict(list)
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            units[row["unit"]].append((int(row["k"]), int(row["bytes"]),
                                        float(row["error"]), float(row.get("weight") or 1)))
    if not units:
        raise ValueError("No profile rows")
    result = []
    for unit, rows in sorted(units.items()):
        rows.sort()
        if len({r[0] for r in rows}) != len(rows) or any(b[1] <= a[1] for a, b in zip(rows, rows[1:])):
            raise ValueError(f"Invalid candidate chain for {unit}")
        if any(r[2] < 0 or r[3] < 0 for r in rows) or len({r[3] for r in rows}) != 1:
            raise ValueError(f"Invalid error or inconsistent weight for {unit}")
        result.append((unit, rows))
    return result


def allocate(units, budget, plus=False):
    spent = sum(rows[0][1] for _, rows in units)
    if spent > budget:
        raise ValueError(f"Minimum placement {spent} exceeds budget {budget}")
    position = [0] * len(units)
    queue = []
    skipped_predecessor = 0
    def push(i, j):
        rows = units[i][1]
        if j + 1 >= len(rows):
            return
        left, right = rows[j:j+2]
        gain = left[3] * max(0.0, left[2] - right[2])
        delta = right[1] - left[1]
        heapq.heappush(queue, (-gain / delta, i, j, delta))
    for i, (_, rows) in enumerate(units):
        for j in range(1 if plus else len(rows)-1):
            push(i, j)
    while queue:
        _, i, j, delta = heapq.heappop(queue)
        if position[i] != j:
            skipped_predecessor += 1
            continue
        if spent + delta > budget:
            continue
        position[i] += 1
        spent += delta
        if plus:
            push(i, position[i])
    total_weight = sum(rows[0][3] for _, rows in units)
    weighted_error = sum(rows[p][2] * rows[p][3] for (_, rows), p in zip(units, position)) / total_weight
    return {"bytes": spent, "weighted_error": weighted_error,
            "skipped_predecessor": skipped_predecessor,
            "selection": {unit: rows[p][0] for (unit, rows), p in zip(units, position)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles")
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    units = load_profiles(args.profiles)
    result = {"source": args.profiles, "budget": args.budget,
              "hasp": allocate(units, args.budget),
              "hasp_plus": allocate(units, args.budget, plus=True)}
    payload = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(payload + "\n")
    else:
        print(payload)


if __name__ == "__main__":
    main()
