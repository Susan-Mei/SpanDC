# Core module entry points

This directory contains four executable entry points for the SpanDC pipeline.
Run any entry point with `--help` to inspect its settings.

1. `01_mining/run.py` mines a RuleStore and serializes its predicate space.
2. `02_query/run.py` queries exact minimal rules for an inclusive interval.
3. `03_segmentation/run.py` selects semantic boundaries and annotates each span.
4. `04_cleaning/run.py` ranks dirty cells by rule-violation participation.

The cleaning entry point consumes a same-schema table and discovered rules,
then writes a ranked list of suspicious cells without changing the input.

Paper figures and tables are reproduced separately under `reproduction/`.
