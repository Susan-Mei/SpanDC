# REPRODUCE

## Objective

Export enhanced abcOD-style rules for SpanDC cleaning validation.

This is not a full abcOD discovery reproduction. It preserves the rule semantics
needed for cleaning: ordered series scopes, increasing/decreasing/stable band
tolerances, detectors, and projection-style repair hints.

## Environment

Python 3.10+ with no extra dependency.

## Data

Use a numeric ordered CSV with at least two selected attributes. The first
selected attribute is treated as the order attribute and the second as the
ordered value.

## Rule Export Command

```bash
python baselines/abcod/export_rules.py \
  --input-csv data/exp/ts_has11_slice100/ts_11_slice100.csv \
  --attrs time_sec x-acc \
  --output results/dev/abcod_smoke/artifacts/json/abcod_rules_smoke.json \
  --row-limit 100 \
  --overwrite
```

## Expected Metrics

TODO. Paper-level metrics require the original paper protocol and datasets.

## Smoke Test

The command should produce a JSON payload with `method = "abcod"` and a `rules`
array.

## Known Issues

- LMB optimization, dynamic-programming piece selection, and full bidirectional
  abcOD discovery are not implemented.
- The exported rules serve the downstream cleaning task only.
