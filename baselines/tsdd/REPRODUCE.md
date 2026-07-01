# REPRODUCE

## Objective

Export enhanced TSDD-style rules for SpanDC cleaning validation.

This is not a full TSDDISCOVER reproduction. It preserves the rule semantics
needed for cleaning: multi-lag same-attribute and cross-attribute lagged
functions, residual tolerances, scoped detectors, and projection-style repair
hints.

## Environment

Python 3.10+ with no extra dependency.

## Data

Use a numeric ordered CSV. Missing values are skipped by the caller.

## Rule Export Command

```bash
python baselines/tsdd/export_rules.py \
  --input-csv data/exp/ts_room/Occupancy.csv \
  --attrs Temperature Humidity CO2 \
  --output results/dev/tsdd_smoke/artifacts/json/tsdd_rules_smoke.json \
  --row-limit 100 \
  --overwrite
```

## Expected Metrics

TODO. Paper-level metrics require the original paper protocol and datasets.

## Smoke Test

The command should produce a JSON payload with `method = "tsdd"` and non-empty
`rules`.

## Known Issues

- General symbolic regression, implication checking, and full pattern
  validation are intentionally not implemented.
- The exported rules serve the downstream cleaning task only.
