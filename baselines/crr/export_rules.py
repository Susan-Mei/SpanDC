#!/usr/bin/env python3
"""Export cleaning-focused minimal CRR rules."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO / "experiment"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from utils.rule_specs import export_rules_from_rows, load_numeric_csv, rules_payload

METHOD = "crr"


def main() -> int:
    args = parse_args()
    attrs = tuple(args.attrs)
    rows = load_numeric_csv(args.input_csv, attrs, args.row_limit)
    rules = export_rules_from_rows(rows, METHOD, attrs, args.max_rules)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rules_payload(METHOD, attrs, rules), indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--attrs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-limit", type=int, default=200)
    parser.add_argument("--max-rules", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
