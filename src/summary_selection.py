"""Label-free endpoint-chord elbow selection."""

import numpy as np


def elbow_selection(curve, max_k: int):
    candidates = sorted((row for row in curve if row["k"] <= max_k), key=lambda row: row["k"])
    if not candidates:
        return {"status": "infeasible", "max_k": max_k}
    error = np.asarray([row["objective"] for row in candidates], dtype=float)
    if not np.isfinite(error).all() or np.any(np.diff(error) > 1e-7):
        raise ValueError("loss curve must be finite and nonincreasing")
    reduction = error[0] - error[-1]
    flat = reduction <= 1e-10 * max(1.0, abs(error[0]))
    if len(candidates) == 1 or flat:
        score = np.zeros(len(candidates))
    else:
        counts = np.asarray([row["k"] for row in candidates])
        score = (error[0] - error) / reduction - (counts - counts[0]) / (counts[-1] - counts[0])
    chosen = int(np.flatnonzero(score >= score.max() - 1e-12)[0])
    return {**candidates[chosen], "status": "completed", "max_k": max_k,
            "elbow_score": float(score[chosen]), "flat_curve": bool(flat)}
