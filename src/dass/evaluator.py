import numpy as np

DEFAULT_QS = [.01,.05,.10,.25,.50,.75,.90,.95,.99]

def tie_aware_rank_interval(sorted_values, value):
    n = len(sorted_values)
    left = int(np.searchsorted(sorted_values, value, side="left"))
    right = int(np.searchsorted(sorted_values, value, side="right"))
    return left / n, right / n

def tie_aware_rank_error(sorted_values, estimated_value, q):
    low, high = tie_aware_rank_interval(sorted_values, estimated_value)
    if low <= q <= high:
        return 0.0
    if q < low:
        return low - q
    return q - high

def evaluate_rank_error(sketch, values, qs=DEFAULT_QS):
    sorted_values = np.sort(np.asarray(values, dtype=float))
    errors = []
    for q in qs:
        estimate = float(sketch.get_quantile(float(q)))
        errors.append(tie_aware_rank_error(sorted_values, estimate, float(q)))
    return {
        "mean_rank_error": float(np.mean(errors)),
        "max_rank_error": float(np.max(errors)),
        "p95_rank_error": float(np.quantile(errors, .95)),
    }

def evaluate_per_quantile(sketch, values, qs=DEFAULT_QS):
    sorted_values = np.sort(np.asarray(values, dtype=float))
    rows = []
    for q in qs:
        estimate = float(sketch.get_quantile(float(q)))
        low, high = tie_aware_rank_interval(sorted_values, estimate)
        rows.append({"q":float(q),"estimate":estimate,"rank_low":low,"rank_high":high,
                     "rank_error":tie_aware_rank_error(sorted_values, estimate, q)})
    return rows
