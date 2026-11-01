import csv

from artifact_io import read_core_artifact
from rulestore import query_rule_store
from spandc import main as mining_main
from spandc_query import main as query_main


def test_mining_and_query_end_to_end(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "series.csv").write_text("x\n0\n1\n2\n3\n", encoding="utf-8")
    metadata = dataset / "DATASET_INFO.toml"
    metadata.write_text(
        """data_file = "series.csv"

[order]
order_keys = []

[attributes]
numeric = ["x"]
character = []

[predicate_space]
candidate_attributes = ["x"]
cross_attribute_comparison_groups = []
""",
        encoding="utf-8",
    )
    artifact = tmp_path / "artifact"
    query_output = tmp_path / "query"
    shared = [
        "--minimum-interval-length", "2",
        "--maximum-predicate-length", "1",
        "--max-spans", "4",
    ]

    assert mining_main(["--dataset-info", str(metadata), "--output", str(artifact), *shared]) == 0
    rule_store, _, stored_metadata = read_core_artifact(artifact)
    assert stored_metadata["row_count"] == 4
    assert query_rule_store(rule_store, 0, 3) == frozenset({8})

    assert query_main([
        "0", "3", "--artifact-dir", str(artifact), "--output", str(query_output), *shared,
    ]) == 0
    with (query_output / "query_rules.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{
        "query_left": "0",
        "query_right": "3",
        "rule_mask": "8",
        "rule_length": "1",
        "rule_body": "t.x > t'.x",
    }]
