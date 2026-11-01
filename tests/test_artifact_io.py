from artifact_io import read_core_artifact, write_core_artifact
from predicates import Predicate
from rulestore import RuleStoreInterval


def test_core_artifact_round_trip(tmp_path):
    predicates = [Predicate(0, "x", "x", ">", "numeric", "x||x", "tdc4", -1.0, 1.0)]
    rule_store = {1: (RuleStoreInterval(2, 9), RuleStoreInterval(12, 20))}
    metadata = {
        "schema_version": 1,
        "row_count": 21,
        "minimum_interval_length": 20,
        "maximum_predicate_length": 3,
        "max_spans": 32,
    }

    write_core_artifact(tmp_path, rule_store, predicates, metadata)
    loaded_store, loaded_predicates, loaded_metadata = read_core_artifact(tmp_path)

    assert loaded_store == rule_store
    assert loaded_predicates == predicates
    assert loaded_metadata == metadata
