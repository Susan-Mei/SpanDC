import sys
from scipy.signal import resample

sys.path.insert(0, "../")

from external.competitor.time2feat import feature_extraction, feature_selection, ClusterWrapper

import os, time

from sklearn.metrics import adjusted_mutual_info_score

import daproli as dp
import pandas as pd
from tqdm import tqdm

from benchmark.metrics import f_measure, covering
from src.clap import CLaP
from src.utils import create_state_labels, load_tssb_datasets, load_datasets, load_has_datasets, extract_cps, \
    load_train_dataset, normalize_time_series

import numpy as np

np.random.seed(1379)


# Runs CLaP experiment
def evaluate_clap(dataset, w, cps, labels, ts, **seg_kwargs):
    seg_df = seg_kwargs["segmentations"]["ClaSP"]
    found_cps = seg_df.loc[seg_df["dataset"] == dataset].iloc[0].found_cps

    runtime = time.process_time()

    clap = CLaP()
    clap.fit(ts, found_cps)

    found_cps = clap.get_change_points()
    found_labels = clap.get_segment_labels()

    true_seg_labels = create_state_labels(cps, labels, ts.shape[0])
    pred_seg_labels = create_state_labels(found_cps, found_labels, ts.shape[0])

    runtime = time.process_time() - runtime
    return evaluate_state_detection_algorithm(dataset, ts.shape[0], cps, found_cps, true_seg_labels, pred_seg_labels,
                                              runtime)


# Runs ClaSP2Feat experiment
def evaluate_clasp2feat(dataset, w, cps, labels, ts, **seg_kwargs):
    seg_df = seg_kwargs["segmentations"]["ClaSP"]

    found_cps = seg_df.loc[seg_df["dataset"] == dataset].iloc[0].found_cps
    found_cps = np.array([0] + found_cps.tolist() + [ts.shape[0]])

    runtime = time.process_time()
    windows = [ts[found_cps[idx]:found_cps[idx + 1]] for idx in range(len(found_cps) - 1)]

    # Create equal length windows
    min_len = np.min([w.shape[0] for w in windows])
    windows = np.array([w[:min_len] for w in windows])

    if ts.ndim == 1: windows = np.array([np.array([w]) for w in windows])

    # Model params
    transform_type = 'minmax'
    model_type = 'Hierarchical'
    context = {'model_type': model_type, 'transform_type': transform_type}

    try:
        df_features = feature_extraction(windows, batch_size=500)
        top_features = feature_selection(df_features, None, context)
        df_features = df_features[top_features]
        model = ClusterWrapper(n_clusters=np.unique(labels).shape[0], model_type=model_type,
                               transform_type=transform_type)
        found_labels = model.fit_predict(df_features.values)
    except Exception as e:
        print(f"Exception: {e}; using only zero class.")
        found_labels = np.zeros(found_cps.shape[0] + 1, dtype=int)

    true_seg_labels = create_state_labels(cps, labels, ts.shape[0])
    pred_seg_labels = create_state_labels(found_cps, found_labels, ts.shape[0])

    runtime = time.process_time() - runtime
    return evaluate_state_detection_algorithm(dataset, ts.shape[0], cps, found_cps, true_seg_labels, pred_seg_labels,
                                              runtime)


# Runs Time2State experiment
def evaluate_time2state(dataset, w, cps, labels, ts, **seg_kwargs):
    from external.competitor.time2state import CausalConv_LSE_Adaper, DPGMM, params_LSE, Time2State
    runtime = time.process_time()

    window_size, step = 256, 10
    params_LSE['in_channels'] = 1 if ts.ndim == 1 else ts.shape[1]

    # min-max normalize ts
    ts = normalize_time_series(ts)

    if ts.ndim == 1:
        ts = ts.reshape(-1, 1)

    true_seg_labels = create_state_labels(cps, labels, ts.shape[0])

    try:
        t2s = Time2State(
            window_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)
        ).fit(ts, window_size, step)

        pred_seg_labels = t2s.state_seq
    except ValueError as e:
        print(f"Exception: {e}; using only zero class.")
        pred_seg_labels = np.zeros_like(true_seg_labels)

    found_cps = extract_cps(pred_seg_labels)
    runtime = time.process_time() - runtime

    return evaluate_state_detection_algorithm(dataset, ts.shape[0], cps, found_cps, true_seg_labels, pred_seg_labels,
                                              runtime)


# Runs E2USD experiment
def evaluate_e2usd(dataset, w, cps, labels, ts, **seg_kwargs):
    from external.competitor.e2usd import E2USD, E2USD_Adaper, DPGMM, params
    runtime = time.process_time()

    window_size, step = 256, 50
    params['in_channels'] = 1 if ts.ndim == 1 else ts.shape[1]

    # min-max normalize ts
    ts = normalize_time_series(ts)

    if ts.ndim == 1:
        ts = ts.reshape(-1, 1)

    true_seg_labels = create_state_labels(cps, labels, ts.shape[0])

    try:
        e2usd = E2USD(
            window_size, step, E2USD_Adaper(params), DPGMM(None)
        ).fit(ts, window_size, step)

        pred_seg_labels = e2usd.state_seq
    except ValueError as e:
        print(f"Exception: {e}; using only zero class.")
        pred_seg_labels = np.zeros_like(true_seg_labels)

    found_cps = extract_cps(pred_seg_labels)
    runtime = time.process_time() - runtime

    return evaluate_state_detection_algorithm(dataset, ts.shape[0], cps, found_cps, true_seg_labels, pred_seg_labels,
                                              runtime)


# Runs TICC experiment
def evaluate_ticc(dataset, w, cps, labels, ts, **seg_kwargs):
    from external.competitor.ticc import TICC
    runtime = time.process_time()

    # From Time2State supporting website
    num_state = np.unique(labels).shape[0]
    lambda_parameter = 1e-3
    beta = 2200
    threshold = 1e-4
    maxIters = 10

    if ts.ndim == 1:
        ts = ts.reshape(-1, 1)

    true_seg_labels = create_state_labels(cps, labels, ts.shape[0] - w + 1)

    try:
        ticc = TICC(window_size=w, number_of_clusters=num_state, lambda_parameter=lambda_parameter, beta=beta,
                    threshold=threshold, maxIters=maxIters)

        pred_seg_labels, _ = ticc.fit_transform(ts)
        pred_seg_labels = pred_seg_labels.astype(np.int64)
    except Exception as e:
        print(f"Exception: {e}; using only zero class.")
        pred_seg_labels = np.zeros_like(true_seg_labels)

    found_cps = extract_cps(pred_seg_labels)
    runtime = time.process_time() - runtime

    return evaluate_state_detection_algorithm(dataset, ts.shape[0], cps, found_cps, true_seg_labels, pred_seg_labels,
                                              runtime)


# Runs AutoPlait experiment
def evaluate_autoplait(dataset, w, cps, labels, ts, **seg_kwargs):
    from external.competitor.autoplait import autoplait

    runtime = time.process_time()
    found_cps, found_labels = autoplait(dataset, ts, cps.shape[0])

    pred_seg_labels = create_state_labels(found_cps, found_labels, ts.shape[0])
    true_seg_labels = create_state_labels(cps, labels, ts.shape[0])
    runtime = time.process_time() - runtime

    return evaluate_state_detection_algorithm(dataset, ts.shape[0], cps, found_cps, true_seg_labels, pred_seg_labels,
                                              runtime)


# Runs HDP-HSMM experiment
def evaluate_hdp_hsmm(dataset, w, cps, labels, ts, **seg_kwargs):
    from external.competitor.hdp_hsmm import HDP_HSMM
    runtime = time.process_time()

    if ts.ndim == 1:
        ts = ts.reshape(-1, 1)

    # HDP-HSMM code breaks for TS with > 200k data points
    if ts.shape[0] > 200_000:
        resample_rate = 200_000 / ts.shape[0]
        ts = resample(ts, int(ts.shape[0] * resample_rate))
        cps = np.array(cps * resample_rate, dtype=int)

    true_seg_labels = create_state_labels(cps, labels, ts.shape[0])

    # From Time2State supporting website
    alpha = 1e4
    beta = 20

    try:
        hdp = HDP_HSMM(alpha=alpha, beta=beta)
        pred_seg_labels = hdp.fit_transform(ts)
    except Exception as e:
        print(f"Exception: {e}; using only zero class.")
        pred_seg_labels = np.zeros_like(true_seg_labels)

    found_cps = extract_cps(pred_seg_labels)
    runtime = time.process_time() - runtime

    return evaluate_state_detection_algorithm(dataset, ts.shape[0], cps, found_cps, true_seg_labels, pred_seg_labels,
                                              runtime)


# Evaluates state detection experiment
def evaluate_state_detection_algorithm(dataset, n_timestamps, cps_true, cps_pred, labels_true, labels_pred, runtime,
                                       verbose=0):
    f1_score = np.round(f_measure({0: cps_true}, cps_pred, margin=int(n_timestamps * .01)), 3)
    covering_score = np.round(covering({0: cps_true}, cps_pred, n_timestamps), 3)
    ami = np.round(adjusted_mutual_info_score(labels_true, labels_pred), 3)

    if verbose > 0:
        print(f"{dataset}: F1-Score: {f1_score}, Covering-Score: {covering_score}, AMI-Score: {ami}")

    return dataset, cps_true.tolist(), cps_pred.tolist(), labels_pred.tolist(), f1_score, covering_score, ami, runtime


# Runs state detection experiment for single algorithm and data set
def evaluate_candidate(dataset_name, candidate_name, eval_func, columns=None, n_jobs=1, verbose=0, **seg_kwargs):
    if dataset_name == "train":
        df = load_train_dataset()
    elif dataset_name == "TSSB":
        df = load_tssb_datasets()
    elif dataset_name == "HAS":
        df = load_has_datasets()
    else:
        df = load_datasets(dataset_name)

    df_cand = dp.map(
        lambda _, args: eval_func(*args, **seg_kwargs),
        tqdm(list(df.iterrows()), disable=verbose < 1),
        ret_type=list,
        verbose=0,
        n_jobs=n_jobs,
    )

    if columns is None:
        columns = ["dataset", "true_cps", "found_cps", "found_labels", "f1_score", "covering_score", "ami_score",
                   "runtime"]

    df_cand = pd.DataFrame.from_records(
        df_cand,
        index="dataset",
        columns=columns,
    )

    print(
        f"{dataset_name}: {candidate_name}: mean_f1_score={np.round(df_cand.f1_score.mean(), 3)}, mean_covering_score={np.round(df_cand.covering_score.mean(), 3)}, mean_ami_score={np.round(df_cand.ami_score.mean(), 3)}")
    return df_cand


# Runs state detection experiment for multiple algorithms and single data set
def evaluate_competitor(dataset_name, exp_path, n_jobs, verbose):
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    competitors = [
        ("CLaP", evaluate_clap),
        ("Time2State", evaluate_time2state),
        ("E2USD", evaluate_e2usd),
        ("TICC", evaluate_ticc),
        ("AutoPlait", evaluate_autoplait),
        ("HDP-HSMM", evaluate_hdp_hsmm),
        ("ClaSP2Feat", evaluate_clasp2feat),
    ]

    # Load segmentations
    segmentations = {}

    for seg_algo in ("ClaSP",):  #
        converters = dict([(column, lambda data: np.array(eval(data))) for column in ["found_cps"]])
        segmentations[seg_algo] = pd.read_csv(
            f"../experiments/segmentation/{dataset_name}_{seg_algo}.csv.gz",
            converters=converters
        )[["dataset", "found_cps"]]

    for candidate_name, eval_func in competitors:
        print(f"Evaluating competitor: {candidate_name}")

        df = evaluate_candidate(
            dataset_name,
            candidate_name,
            eval_func=eval_func,
            columns=None,
            n_jobs=n_jobs,
            verbose=verbose,
            segmentations=segmentations
        )

        df.to_csv(f"{exp_path}{dataset_name}_{candidate_name}.csv.gz", compression='gzip')


if __name__ == '__main__':
    exp_path = "../experiments/state_detection/"
    n_jobs, verbose = 1, 0

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    for bench in ("TSSB", "UTSA", "SKAB", "HAS", "MIT-BIH"):
        evaluate_competitor(bench, exp_path, n_jobs, verbose)
