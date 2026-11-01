# Dataset acquisition and local layout

The paper uses eight public dataset families. Upstream licenses and access terms
remain authoritative; this repository does not redistribute their raw files.

| Paper family | Upstream | Acquisition | Local converted file |
| --- | --- | --- | --- |
| HAS | [Human Activity Segmentation Challenge](https://www.kaggle.com/competitions/human-activity-segmentation-challenge/data) | Accept the Kaggle competition terms, then run `kaggle competitions download -c human-activity-segmentation-challenge` | one ordered CSV per selected sequence, e.g. `data/raw/ts_has11/ts_11.csv` |
| MoCap | [CMU motion segmentation data](https://graphics.cs.cmu.edu/projects/segmentation/) | Download the 14-sequence measurements and ground-truth key from the project page | converted selected records such as `data/raw/ts_mocap_5/ts_5.csv` |
| Art | [UCI Daily and Sports Activities](https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities) | Download the UCI archive; preserve subject/activity/time order | `data/raw/ts_art/data_labeled.csv` |
| Occupancy | [UCI Occupancy Detection](https://archive.ics.uci.edu/dataset/357/occupancy+detection) | Download and unzip the three supplied CSV/TXT splits | `data/raw/ts_room/Occupancy.csv` |
| Telemetry | [Environmental Sensor Telemetry Data](https://www.kaggle.com/datasets/garystafford/environmental-sensor-data-132k) | `kaggle datasets download -d garystafford/environmental-sensor-data-132k` | `data/raw/ts_environment/iot_telemetry_data.csv` |
| Gas | [UCI Gas sensor array temperature modulation](https://archive.ics.uci.edu/dataset/487/gas+sensor+array+temperature+modulation) | Download the UCI ZIP; concatenate source CSVs in filename order with one header | `data/raw/ts_gas/ts_gas_merged.csv` |
| HEPC | [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) | Download the ZIP, parse the semicolon file, keep chronological order, and omit rows with missing selected measurements | `data/raw/ts_hepc/household_power_consumption.csv` |
| Fantasia | [PhysioNet Fantasia 1.0.0](https://physionet.org/content/fantasia/1.0.0/) | `wget -r -N -c -np https://physionet.org/files/fantasia/1.0.0/`; convert the selected WFDB record without resampling | `data/raw/ts_fantasia/f1y05.csv` |

All declared measurement columns are included in predicate construction.
Labels, timestamps, row identifiers, and change-point flags remain excluded.

Every local CSV must preserve source order and full numeric precision. Its
`DATASET_INFO.toml` declares:

- `data_file` and ordering columns;
- numeric/categorical attributes;
- the exact candidate attributes and cross-attribute comparison groups;
- labels retained as metadata but excluded from mining.

`data/views/` contains deterministic sample views for quick local runs. Each
view is a prefix or a documented ordered slice of its source dataset.
