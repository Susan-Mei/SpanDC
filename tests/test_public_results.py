import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_paper_results_are_csv_only_and_have_no_speculative_fields():
    files = sorted((ROOT / "results" / "paper").iterdir())
    assert files and all(path.is_file() and path.suffix == ".csv" for path in files)
    forbidden = ("estimate", "prediction", "predicted", "forecast", "speculat", "extrapolat")
    for path in files:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            assert header
            assert not any(token in field.lower() for field in header for token in forbidden)
            assert all(len(row) == len(header) for row in reader)
