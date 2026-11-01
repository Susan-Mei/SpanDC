import math

from cleaning import rank_dirty_cells
from predicates import Predicate


def test_violation_participation_returns_ranked_cells():
    rows = [{"x": 3.0}, {"x": 2.0}, {"x": 1.0}]
    predicates = [Predicate(0, "x", "x", ">", "numeric", "x||x")]
    scoped_rules = [{"start": 0, "end_exclusive": 3, "rule_mask": 1}]

    ranked, diagnostics = rank_dirty_cells(rows, predicates, scoped_rules)

    assert diagnostics[0]["violations"] == 3
    assert [(item["row_index"], item["attribute"]) for item in ranked] == [(0, "x"), (1, "x"), (2, "x")]
    fanout = 3 / 3
    expected = 2 * math.log1p(3) / (1 + fanout) * (1 + 1 / (1 + fanout))
    assert all(math.isclose(item["violation_score"], expected) for item in ranked)
    assert all(set(item) == {"row_index", "attribute", "violation_score"} for item in ranked)
