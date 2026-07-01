# SpanDC Artifact

This repository contains the SpanDC source code and the paper-facing artifact
data needed to reproduce the experiment tables and figures.

## Setup

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[plot]"
```

## Reproduce Outputs

```bash
python experiment/reproduce_all.py
```

Generated files are written under:

```text
results/reproduced/
```

The default command rebuilds rule-quality table, cleaning 
figure, runtime/storage table, and scalability figures.

## Run SpanDC Directly

After setup and raw dataset placement, run the packaged SpanDC showcase command:

```bash
python -m spandc
```

This runs SpanDC on MoCap-5 , maximum predicate length 3,
It writes:

```text
results/reproduced/spandc_demo/rulestore.csv
results/reproduced/spandc_demo/interval_rules.csv
```

The same entry point is also available as:

```bash
spandc
```

## Query RuleStore Rules

To query minimal valid rules on a interval, pass the inclusive left and
right tuple indices:

```bash
python -m spandc_query 0 100
```

This writes:

```text
results/reproduced/spandc_query/query_rules.csv
```

The installed command is:

```bash
spandc-query 0 100
```

## Raw Dataset Sources

Download the upstream datasets below and place or convert the listed local files
under `data/raw/<dataset_id>/`.

| Local dataset | Upstream source | Download entry |
| --- | --- | --- |
| `ts_art` | UCI Daily and Sports Activities | https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities |
| `ts_climate` | Daily Climate time series data | `kaggle datasets download -d sumanthvrao/daily-climate-time-series-data` |
| `ts_environment` | Environmental Sensor Telemetry Data | `kaggle datasets download -d garystafford/environmental-sensor-data-132k` |
| `ts_gas` | Gas sensor array temperature modulation | `kaggle datasets download -d javi2270784/gas-sensor-array-temperature-modulation` |
| `ts_has11` | Human Activity Segmentation Challenge @ ECML/PKDD 2023 | https://www.kaggle.com/competitions/human-activity-segmentation-challenge/data |
| `ts_mocap_5`, `ts_mocap_7` | CMU motion-capture segmentation data | https://graphics.cs.cmu.edu/projects/segmentation/ |
| `ts_pump` | Pump sensor data | `kaggle datasets download -d nphantawee/pump-sensor-data` |
| `ts_room` | UCI Occupancy Detection | https://archive.ics.uci.edu/dataset/357/occupancy+detection |

The local file expected by each dataset is specified in
`data/raw/<dataset_id>/DATASET_INFO.toml` under `data_file`.

## Layout

```text
src/                  SpanDC core source code.
experiment/           Paper artifact modules.
baselines/            Baseline adapters used by the artifact modules.
data/raw/             Raw datasets used by paper experiments.
data/views/           Deterministic paper views.
results/paper/        Checked-in paper data and reference figures.
```
