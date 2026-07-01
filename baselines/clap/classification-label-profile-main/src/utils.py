import os

ABS_PATH = os.path.dirname(os.path.abspath(__file__))

from numba import njit

import numpy as np
import pandas as pd


# Loads specified data set
def load_datasets(dataset, selection=None):
    desc_filename = ABS_PATH + f"/../datasets/{dataset}/desc.txt"
    desc_file = []

    with open(desc_filename, 'r') as file:
        for line in file.readlines(): desc_file.append(line.split(","))

    prop_filename = ABS_PATH + f"/../datasets/{dataset}/properties.txt"
    prop_file = []

    with open(prop_filename, 'r') as file:
        for line in file.readlines(): prop_file.append(line.split(","))

    assert len(desc_file) == len(prop_file), "Description and property file have different records."

    df = []

    for idx, (desc_row, prop_row) in enumerate(zip(desc_file, prop_file)):
        if selection is not None and idx not in selection: continue
        assert desc_row[0] == prop_row[0], f"Description and property row {idx} have different records."

        (ts_name, window_size), change_points = desc_row[:2], desc_row[2:]
        labels = prop_row[1:]

        if len(change_points) == 1 and change_points[0] == "\n": change_points = list()
        path = ABS_PATH + f'/../datasets/{dataset}/'

        if os.path.exists(path + ts_name + ".txt"):
            ts = np.loadtxt(fname=path + ts_name + ".txt", dtype=np.float64)
        else:
            ts = np.load(file=path + "data.npz")[ts_name]

        df.append((ts_name, int(window_size), np.array([int(_) for _ in change_points]),
                   np.array([int(_) for _ in labels]), ts))

    return pd.DataFrame.from_records(df, columns=["dataset", "window_size", "change_points", "labels", "time_series"])


# Loads TSSB data set
def load_tssb_datasets(names=None):
    desc_filename = os.path.join(ABS_PATH, "../datasets/TSSB", "desc.txt")
    desc_file = []

    with open(desc_filename, 'r') as file:
        for line in file.readlines():
            line = line.split(",")

            if names is None or line[0] in names:
                desc_file.append(line)

    prop_filename = os.path.join(ABS_PATH, "../datasets/TSSB", "properties.txt")
    prop_file = []

    with open(prop_filename, 'r') as file:
        for line in file.readlines():
            line = line.split(",")

            ds_name, interpretable, label_cut, resample_rate, labels = line[0], bool(line[1]), int(line[2]), int(
                line[3]), line[4:]
            labels = [int(l.replace("\n", "")) // (label_cut + 1) for l in labels]

            if names is None or ds_name in names:
                prop_file.append((ds_name, label_cut, resample_rate, labels))

    df = []

    for desc_row, prop_row in zip(desc_file, prop_file):
        (ts_name, window_size), change_points = desc_row[:2], desc_row[2:]
        labels = prop_row[3]

        ts = np.loadtxt(fname=os.path.join(ABS_PATH, "../datasets/TSSB", ts_name + '.txt'), dtype=np.float64)

        df.append((ts_name, int(window_size), np.array([int(_) for _ in change_points]), np.array(labels), ts))

    return pd.DataFrame.from_records(df, columns=["dataset", "window_size", "change_points", "labels", "time_series"])


# Loads HAS data set
def load_has_datasets(selection=None):
    data_path = ABS_PATH + "/../datasets/has2023_master.csv.zip"

    np_cols = ["change_points", "activities", "x-acc", "y-acc", "z-acc",
               "x-gyro", "y-gyro", "z-gyro",
               "x-mag", "y-mag", "z-mag",
               "lat", "lon", "speed"]

    converters = {
        col: lambda val: np.array([]) if len(val) == 0 else np.array(eval(val)) for col
        in np_cols}

    df_has = pd.read_csv(data_path, converters=converters, compression="zip")

    df = []
    sample_rate = 50

    for _, row in df_has.iterrows():
        if selection is not None and row.ts_challenge_id not in selection: continue
        ts_name = f"{row.group}_subject{row.subject}_routine{row.routine} (id{row.ts_challenge_id})"

        label_mapping = {label: idx for idx, label in enumerate(np.unique(row.activities))}
        labels = np.array([label_mapping[label] for label in row.activities])

        if row.group == "indoor":
            ts = np.hstack((
                row["x-acc"].reshape(-1, 1),
                row["y-acc"].reshape(-1, 1),
                row["z-acc"].reshape(-1, 1),
                row["x-gyro"].reshape(-1, 1),
                row["y-gyro"].reshape(-1, 1),
                row["z-gyro"].reshape(-1, 1),
                row["x-mag"].reshape(-1, 1),
                row["y-mag"].reshape(-1, 1),
                row["z-mag"].reshape(-1, 1)
            ))
        elif row.group == "outdoor":
            ts = np.hstack((
                row["x-acc"].reshape(-1, 1),
                row["y-acc"].reshape(-1, 1),
                row["z-acc"].reshape(-1, 1),
                row["x-mag"].reshape(-1, 1),
                row["y-mag"].reshape(-1, 1),
                row["z-mag"].reshape(-1, 1),
            ))
        else:
            raise ValueError("Unknown group in HAS dataset.")

        df.append((ts_name, sample_rate, row.change_points, labels, ts))

    if selection is None:
        selection = np.arange(df_has.shape[0])

    return pd.DataFrame.from_records(
        df,
        columns=["dataset", "window_size", "change_points", "labels", "time_series"]
    ).iloc[selection, :]


# Loads train data set (for ablations)
def load_train_dataset():
    train_names = [
        'DodgerLoopDay',
        'EEGRat',
        'EEGRat2',
        'FaceFour',
        'GrandMalSeizures2',
        'GreatBarbet1',
        'Herring',
        'InlineSkate',
        'InsectEPG1',
        'MelbournePedestrian',
        'NogunGun',
        'NonInvasiveFetalECGThorax1',
        'ShapesAll',
        'TiltECG',
        'ToeSegmentation1',
        'ToeSegmentation2',
        'Trace',
        'UWaveGestureLibraryY',
        'UWaveGestureLibraryZ',
        'WordSynonyms',
        'Yoga'
    ]

    df = pd.concat([load_datasets("UTSA"), load_tssb_datasets()])
    df = df[df["dataset"].isin(train_names)]

    return df.sort_values(by="dataset")


# Normalizes multivariate time series
def normalize_time_series(ts):
    flatten = False

    if ts.ndim == 1:
        ts = ts.reshape(-1, 1)
        flatten = True

    for dim in range(ts.shape[1]):
        channel = ts[:, dim]

        # Min-max normalize channel
        try:
            channel = np.true_divide(channel - channel.min(), channel.max() - channel.min())
        except FloatingPointError:
            pass

        # Interpolate (if missing values are present)
        channel[np.isinf(channel)] = np.nan
        channel = pd.Series(channel).interpolate(limit_direction="both").to_numpy()

        # There are series that still contain NaN values
        channel[np.isnan(channel)] = 0

        ts[:, dim] = channel

    if flatten:
        ts = ts.flatten()

    return ts


# Create vector of state labels that map to data points
@njit(fastmath=True, cache=True)
def create_state_labels(cps, labels, ts_len):
    seg_labels = np.zeros(shape=ts_len, dtype=np.int64)

    segments = np.concatenate((
        np.array([0]),
        cps,
        np.array([ts_len])
    ))

    for idx in range(1, len(segments)):
        seg_start, seg_end = segments[idx - 1], segments[idx]
        seg_labels[seg_start:seg_end] = labels[idx - 1]

    return seg_labels


# Creates a sliding window from a time series
def create_sliding_window(time_series, window_size, stride=1):
    X = []

    for idx in range(0, time_series.shape[0], stride):
        if idx + window_size <= time_series.shape[0]:
            X.append(time_series[idx:idx + window_size])

    return np.array(X, dtype=time_series.dtype)


# Expands a label sequence from a sliding window
def expand_label_sequence(labels, window_size, stride):
    X = []

    for label in labels:
        X.extend([label] * (window_size - (window_size - stride)))

    return np.array(X, dtype=labels.dtype)


# Collapses a label sequence to its dense representation
def collapse_label_sequence(label_seq):
    labels = []

    for idx in range(1, len(label_seq)):
        if label_seq[idx - 1] != label_seq[idx]:
            labels.append(label_seq[idx - 1])

        if idx == len(label_seq) - 1:
            labels.append(label_seq[idx])

    return np.array(labels)


# Extracts CPs from a label sequence
def extract_cps(label_seq):
    label_diffs = label_seq[:-1] != label_seq[1:]
    return np.arange(label_seq.shape[0] - 1)[label_diffs] + 1
