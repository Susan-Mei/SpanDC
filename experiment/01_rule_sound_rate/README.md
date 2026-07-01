# 01 Rule Sound Rate

This module reproduces the rule-quality table from checked-in paper data. The
formal protocol compares FDCD-backed global tDC discovery with SpanDC fixed-k
local discovery on four paper views:

- HAS-11: last six attributes, with the three magnetometer attributes mutually comparable.
- MoCap-5: default paper view.
- MoCap-7: default paper view.
- Occupancy: first 1000 rows, with `Occupancy` kept as a candidate attribute and `K=4`.

The paper data are stored in:

```text
results/paper/01_rule_sound_rate/data/
```

Reproduce the table:

```bash
python experiment/01_rule_sound_rate/report.py
```

Outputs are written to:

```text
results/reproduced/01_rule_sound_rate/
```

The checked-in formal table reports global tDCs for `d_max=2,3,4,5,6` and
SpanDC for `d_max=2,3`. The SpanDC run uses fixed-k segmentation: HAS-11,
MoCap-5, and MoCap-7 use the number of contiguous activity segments; Occupancy
uses `K=4`.

The full formal rebuild is deliberately not executed by `reproduce_all.py`
because it reruns mining for every dataset and length setting. The checked-in
CSV/JSON files are the release source for this table.
