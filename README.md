# SpanDC

This repository releases the SpanDC core for span-local temporal denial
constraints: mining, interval query, semantic segmentation, and rule-based dirty
cell localization. A separate reproduction module reproduces the paper results.

## Install

Python 3.11 is the reference environment.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

`pyroaring` is an optional mining acceleration backend:
`python -m pip install -e ".[accel]"`.

## Four core modules

The scripts below form a pipeline. Their default paths use the bundled
1,000-row Occupancy view; `--row-limit 100` keeps this first run small.
The shared pipeline parameters are available from each entry point's `--help`.
Mining records them in `artifact.json`; query, segmentation, and cleaning
validate that they match before running.

### 01 Mining

```bash
python experiment/01_mining/run.py \
  --dataset-info data/views/ts_room_first1000/DATASET_INFO.toml \
  --row-limit 100
```

Mining writes `results/reproduced/01_mining/`:

- `predicates.csv`: the fixed predicate space and tDC thresholds;
- `rulestore.csv`: each rule's inclusion-maximal minimal-validity intervals;
- `artifact.json`: row count and mining configuration needed downstream.

### 02 Query

```bash
python experiment/02_query/run.py 0 79
```

The query interval is inclusive. The output
`results/reproduced/02_query/query_rules.csv` contains the exact subset-minimal
rules whose RuleStore intervals cover the requested interval.

### 03 Segmentation

```bash
python experiment/03_segmentation/run.py
```

Segmentation builds fixed-width RuleStore profiles, selects boundaries with
nested greedy splitting and elbow selection, and queries exact rules for each
final span. It writes boundaries, interval rules, and the summary curve to
`results/reproduced/03_segmentation/`.

### 04 Cleaning

```bash
python experiment/04_cleaning/run.py \
  --dataset-info data/views/ts_room_first1000/DATASET_INFO.toml \
  --row-limit 100
```

For a same-schema dirty table, pass `--input-csv /path/to/dirty.csv`. The module
loads the fixed predicates and segment rules, counts each cell's participation
in violating tuple pairs, and writes the ranked `cell_scores.csv`. Input values
are left unchanged.

Installed command aliases are `spandc-mine` (also `spandc`), `spandc-query`,
`spandc-segment`, and `spandc-clean`.

## Paper-figure reproduction

Install the plotting dependency and run:

```bash
python -m pip install -e ".[plot]"
python reproduction/reproduce_all.py
```

## Data

The repository includes a small Occupancy view for an end-to-end core run. The
other raw datasets are not redistributed. Follow [data/README.md](data/README.md)
for current upstream locations, access restrictions, expected filenames, and
conversion notes.

## Repository layout

```text
src/                  SpanDC mining, RuleStore, query, boundary, and cleaning code
experiment/           four executable core-module entry points
reproduction/         paper figure and table reproduction
data/raw/              dataset metadata for user-supplied raw files
data/views/            bundled deterministic sample view
results/paper/         CSV-only observed paper data
results/reproduced/    generated core/reproduction outputs (gitignored)
```
