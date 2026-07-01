# 02 Cleaning

This module reproduces the detection-effectiveness figure from checked-in
paper data. The experiment uses the rule sets from `01_rule_sound_rate`:
SpanDC uses `d_max=3` local rules, and global tDC uses `d_max=6` full-sequence
rules. The comparison also includes TSDD, abcOD, and CRR.

The complete four-view metrics are stored in:

```text
results/paper/02_cleaning/data/
```

The paper figure uses MoCap-7 and Occupancy to keep the main text compact:

```bash
python experiment/02_cleaning/plot.py
```

Outputs are written to:

```text
results/reproduced/02_cleaning/
```

The source data file contains these columns:

```text
case_id,dataset,dataset_label,error_rate,method,method_label,seed_count,completed_count,f1_mean,f1_std,auprc_mean,auprc_std
```

The experiment reports F1 and AUPRC only. Dirty cells are shared across methods
under a single cell perturbation protocol, so method comparisons are not
confounded by different corrupted inputs.

The full formal rebuild is deliberately not executed by `reproduce_all.py`
because it reruns rule compilation, perturbation, and all detector evaluations.
The checked-in CSV files are the release source for this figure.
