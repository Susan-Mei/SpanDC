from predicates import Predicate
from rulestore import RuleStoreInterval
from semantic_summary import SummaryConfig, summarize_rule_store


def test_flat_profiles_choose_one_span_and_exact_annotation():
    predicates = [Predicate(0, "x", "x", "=", "numeric", "x||x")]
    store = {1: (RuleStoreInterval(0, 99),)}
    result = summarize_rule_store(store, predicates, 100, SummaryConfig(max_spans=8))
    assert result.boundaries == ()
    assert result.spans == ((0, 100),)
    assert result.rules_by_span == (frozenset({1}),)
    assert result.curve[0]["k"] == 1


def test_empty_vocabulary_is_a_valid_one_span_summary():
    result = summarize_rule_store({}, [], 100, SummaryConfig())
    assert result.selected_k == 1
    assert result.rules_by_span == (frozenset(),)
