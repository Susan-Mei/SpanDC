# Paper-figure reproduction

Run:

```bash
python reproduction/reproduce_all.py
```

This module reads the checked-in observed tables in `results/paper/` and writes
paper-style PDF/PNG outputs under `results/reproduced/`:

- `02_segmentation`: the continuous two-row heatmap layout of Figure 6;
- `03_cost_sensitivity`: the grouped-bar layout of Figure 12;
- `03_summary_sensitivity`: the three-panel grouped-bar layout of Figure 13;
- `04_stability`: the booktabs-style layout of Table 4.

The rule-quality table is copied as `01_rule_quality.csv`. Figure and table
captions are added by the manuscript source.
