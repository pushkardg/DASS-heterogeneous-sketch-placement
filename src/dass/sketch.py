from datasketches import kll_doubles_sketch
import numpy as np

def build_kll(values, k):
    s = kll_doubles_sketch(k)
    for x in values:
        s.update(float(x))
    return s

def serialized_bytes(sketch):
    try:
        return int(sketch.get_serialized_size_bytes())
    except Exception:
        return len(sketch.serialize())

def empirical_rank(sorted_values, x):
    return float(np.searchsorted(sorted_values, x, side="right") / len(sorted_values))
