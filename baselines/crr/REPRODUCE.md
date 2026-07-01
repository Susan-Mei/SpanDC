# REPRODUCE

## Objective

Export enhanced CRR-style rules for SpanDC cleaning validation.

This is not a full CRR discovery reproduction. It preserves the rule semantics
needed for cleaning: global/window/value-bin conditions, regression functions,
tolerances, detectors, and projection-style repair hints.

## Environment

Python 3.10+ with no extra dependency.

## Data

Use a numeric ordered CSV with at least two selected attributes.

## Rule Export Command

```bash
python baselines/crr/export_rules.py \
  --input-csv data/exp/ts_room/Occupancy.csv \
  --attrs Temperature Humidity CO2 \
  --output results/dev/crr_smoke/artifacts/json/crr_rules_smoke.json \
  --row-limit 100 \
  --overwrite
```

## Expected Metrics

TODO. Paper-level metrics require the original paper protocol and datasets.

## Smoke Test

The command should produce a JSON payload with `method = "crr"` and non-empty
`rules`.

## Known Issues

- Full DNF inference, fusion, translation, and completeness analysis are not
  implemented.
- The exported rules serve the downstream cleaning task only.
