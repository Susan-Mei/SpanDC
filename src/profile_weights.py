"""Fixed rule-family weights for semantic-summary profiles."""

import numpy as np

from rule_segmentation import rule_weights


def profile_rule_weights(masks, predicates, valid):
    """Keep rules that vary across sampled windows and normalize family weights."""
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2 or valid.shape[0] != len(masks) or valid.shape[1] == 0:
        raise ValueError("invalid validity profiles")
    weights = rule_weights(masks, predicates)
    weights *= np.any(valid != valid[:, :1], axis=1)
    if weights.sum():
        weights /= weights.sum()
    return weights
