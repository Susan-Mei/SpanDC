import sys

sys.path.insert(0, "../")

from benchmark.segmentation_test import evaluate_clasp

from benchmark.state_detection_test import evaluate_candidate, evaluate_state_detection_algorithm

import os
from scipy.signal import resample

from src.clap import CLaP
from src.utils import create_state_labels

import numpy as np

np.random.seed(1379)


# Runs CLaP experiment
def evaluate_clap(dataset, w, cps, labels, ts, **seg_kwargs):
    if "resample_rate" in seg_kwargs:
        rate = seg_kwargs["resample_rate"]

        if rate != 1.0:
            ts = resample(ts, int(ts.shape[0] * rate))
            cps = np.array(cps * rate, dtype=int)

    if "noise_level" in seg_kwargs:
        sigma = seg_kwargs["noise_level"]

        if sigma != 0:
            if ts.ndim == 1:
                ts = (ts - np.mean(ts)) / np.std(ts)
            else:
                ts = np.array([(dim - np.mean(dim)) / np.std(dim) for dim in ts.T])

            ts += np.random.normal(0, sigma, size=ts.shape)

    _, _, found_cps, _, _, _ = evaluate_clasp(dataset, w, cps, labels, ts)

    clap = CLaP(n_jobs=4)
    clap.fit(ts, np.array(found_cps))

    found_cps = clap.get_change_points()
    found_labels = clap.get_segment_labels()

    true_seg_labels = create_state_labels(cps, labels, ts.shape[0])
    pred_seg_labels = create_state_labels(found_cps, found_labels, ts.shape[0])

    return evaluate_state_detection_algorithm(dataset, ts.shape[0], cps, found_cps, true_seg_labels, pred_seg_labels,
                                              None)


# Runs resample rate sensitivity experiment
def evaluate_resample_rates(dataset_name, exp_path, n_jobs, verbose):
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    resample_rates = np.arange(.1, 1.9 + 0.1, 0.1)

    for rate in resample_rates:
        candidate_name = f"{round(rate, 1)}_resampled"
        print(f"Evaluating competitor: {candidate_name}")

        df = evaluate_candidate(
            dataset_name,
            candidate_name,
            eval_func=evaluate_clap,
            columns=None,
            n_jobs=n_jobs,
            verbose=verbose,
            resample_rate=rate,
            classifier=candidate_name
        )

        df.to_csv(f"{exp_path}{dataset_name}_{candidate_name}.csv.gz", compression='gzip')


# Runs noise levels sensitivity experiment
def evaluate_noise_levels(dataset_name, exp_path, n_jobs, verbose):
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    noise_levels = np.arange(0, 2.0 + 0.1, 0.1)

    for level in noise_levels:
        candidate_name = f"{round(level, 1)}_noise"
        print(f"Evaluating competitor: {candidate_name}")

        df = evaluate_candidate(
            dataset_name,
            candidate_name,
            eval_func=evaluate_clap,
            columns=None,
            n_jobs=n_jobs,
            verbose=verbose,
            noise_level=level,
            classifier=candidate_name
        )

        df.to_csv(f"{exp_path}{dataset_name}_{candidate_name}.csv.gz", compression='gzip')


if __name__ == '__main__':
    exp_path = "../experiments/sensitivity/"
    n_jobs, verbose = 1, 0

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    evaluate_resample_rates("train", exp_path, n_jobs, verbose)
    evaluate_noise_levels("train", exp_path, n_jobs, verbose)
