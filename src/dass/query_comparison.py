import json
import os
import time

import numpy as np

from .sketch import build_kll
from .storage_pipeline import _iter_values


def _load_catalog(path):
    with open(path) as f:
        return json.load(f)


def _write_uniform_catalog(inputs, output_dir, k, epoch_size=100000, value_column='value'):
    os.makedirs(output_dir, exist_ok=True)
    entries = []
    for stream, path in inputs.items():
        carry = np.empty(0, dtype=float)
        epoch = 0
        for chunk in _iter_values(path, value_column, epoch_size):
            x = np.concatenate([carry, chunk])
            start = 0
            while len(x) - start >= epoch_size:
                part = x[start:start + epoch_size]
                start += epoch_size
                sk = build_kll(part, k)
                blob = bytes(sk.serialize())
                fname = f'{stream}-epoch-{epoch:05d}-k{k}.kll'
                with open(os.path.join(output_dir, fname), 'wb') as f:
                    f.write(blob)
                entries.append({'stream': stream, 'epoch': epoch, 'k': k,
                                'bytes': len(blob), 'file': fname})
                epoch += 1
            carry = x[start:]
        if len(carry) >= 100:
            sk = build_kll(carry, k)
            blob = bytes(sk.serialize())
            fname = f'{stream}-epoch-{epoch:05d}-k{k}.kll'
            with open(os.path.join(output_dir, fname), 'wb') as f:
                f.write(blob)
            entries.append({'stream': stream, 'epoch': epoch, 'k': k,
                            'bytes': len(blob), 'file': fname})
    manifest = {'uniform_k': int(k), 'total_persisted_bytes': sum(e['bytes'] for e in entries),
                'entries': entries}
    with open(os.path.join(output_dir, 'catalog.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _timed_query(catalog_path, stream, quantiles, warmup, repeats):
    from datasketches import kll_doubles_sketch

    catalog = _load_catalog(catalog_path)
    root = os.path.dirname(catalog_path)
    entries = [e for e in catalog['entries'] if e['stream'] == stream]

    def once(timed):
        read_ms = deserialize_ms = merge_ms = quantile_ms = 0.0
        merged = None
        bytes_read = 0
        total0 = time.perf_counter()
        for e in entries:
            t0 = time.perf_counter()
            with open(os.path.join(root, e['file']), 'rb') as f:
                blob = f.read()
            t1 = time.perf_counter()
            sk = kll_doubles_sketch.deserialize(blob)
            t2 = time.perf_counter()
            if merged is None:
                merged = sk
            else:
                merged.merge(sk)
            t3 = time.perf_counter()
            bytes_read += len(blob)
            if timed:
                read_ms += (t1 - t0) * 1000
                deserialize_ms += (t2 - t1) * 1000
                merge_ms += (t3 - t2) * 1000
        tq = time.perf_counter()
        values = [float(merged.get_quantile(float(q))) for q in quantiles]
        tdone = time.perf_counter()
        if timed:
            quantile_ms = (tdone - tq) * 1000
        return {'total_ms': (tdone - total0) * 1000, 'read_ms': read_ms,
                'deserialize_ms': deserialize_ms, 'merge_ms': merge_ms,
                'quantile_ms': quantile_ms, 'bytes_read': bytes_read,
                'values': values}

    for _ in range(warmup):
        once(False)
    samples = [once(True) for _ in range(repeats)]

    def stats(field):
        a = np.asarray([s[field] for s in samples], dtype=float)
        return {'mean': float(a.mean()), 'p50': float(np.quantile(a, .50)),
                'p95': float(np.quantile(a, .95)), 'p99': float(np.quantile(a, .99))}

    return {'stream': stream, 'sketches_read': len(entries),
            'bytes_read': samples[-1]['bytes_read'], 'warmup': warmup, 'repeats': repeats,
            'quantiles': list(quantiles), 'values': samples[-1]['values'],
            'latency_ms': {field: stats(field) for field in
                           ('total_ms', 'read_ms', 'deserialize_ms', 'merge_ms', 'quantile_ms')}}


def matched_query_benchmark(inputs, mixed_catalog_path, output_dir, epoch_size=100000,
                            value_column='value', candidate_ks=(64,128,256,512,1024),
                            quantiles=(0.5,0.9,0.95,0.99), warmup=100, repeats=1000):
    """Compare a persisted mixed-K catalog with the largest feasible uniform K.

    The mixed catalog's actual persisted byte count is the storage ceiling. Uniform
    candidates are physically materialized, and the largest K whose actual bytes do
    not exceed that ceiling is selected. Timings are warm local-file measurements;
    this function does not flush OS caches or claim cold-device performance.
    """
    os.makedirs(output_dir, exist_ok=True)
    mixed = _load_catalog(mixed_catalog_path)
    mixed_bytes = sum(int(e['bytes']) for e in mixed['entries'])
    mixed_streams = sorted({e['stream'] for e in mixed['entries']})
    if set(mixed_streams) != set(inputs):
        raise ValueError('mixed catalog streams must exactly match --inputs streams')

    candidates = []
    for k in candidate_ks:
        d = os.path.join(output_dir, f'uniform-k{k}')
        manifest = _write_uniform_catalog(inputs, d, int(k), epoch_size, value_column)
        candidates.append({'k': int(k), 'bytes': int(manifest['total_persisted_bytes']),
                           'catalog': os.path.join(d, 'catalog.json')})
    feasible = [c for c in candidates if c['bytes'] <= mixed_bytes]
    if not feasible:
        raise ValueError(f'no uniform candidate fits mixed catalog budget of {mixed_bytes} bytes')
    chosen = max(feasible, key=lambda c: c['k'])

    rows = []
    for stream in mixed_streams:
        m = _timed_query(mixed_catalog_path, stream, quantiles, warmup, repeats)
        u = _timed_query(chosen['catalog'], stream, quantiles, warmup, repeats)
        mt = m['latency_ms']['total_ms']['mean']
        ut = u['latency_ms']['total_ms']['mean']
        rows.append({'stream': stream, 'mixed': m, 'uniform': u,
                     'mean_total_overhead_pct': ((mt / ut) - 1.0) * 100.0,
                     'mean_total_ratio_x': mt / ut})

    result = {
        'methodology': {
            'budget_definition': 'actual persisted bytes of supplied mixed-K catalog',
            'uniform_selection': 'largest candidate K whose physically materialized catalog does not exceed mixed-K bytes',
            'cache_regime': 'warm local filesystem after explicit warmup; OS caches are not flushed',
            'timed_path': 'file read -> deserialize -> merge -> four quantile calls',
            'stage_timing_note': 'stage timers are instrumentation and may add small overhead; total_ms is the primary latency metric',
            'warmup': int(warmup), 'repeats': int(repeats),
            'candidate_ks': [int(k) for k in candidate_ks],
        },
        'mixed_catalog': mixed_catalog_path,
        'mixed_total_bytes': int(mixed_bytes),
        'uniform_candidates': candidates,
        'selected_uniform_k': int(chosen['k']),
        'selected_uniform_bytes': int(chosen['bytes']),
        'budget_slack_bytes': int(mixed_bytes - chosen['bytes']),
        'streams': rows,
    }
    with open(os.path.join(output_dir, 'matched_query_comparison.json'), 'w') as f:
        json.dump(result, f, indent=2)
    return result
