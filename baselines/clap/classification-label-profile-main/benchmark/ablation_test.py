import sys

sys.path.insert(0, "../")

from benchmark.state_detection_test import evaluate_candidate, evaluate_state_detection_algorithm

import os

import pandas as pd

from src.clap import CLaP
from src.utils import create_state_labels

import numpy as np

np.random.seed(1379)


# Runs CLaP experiment
def evaluate_clap(dataset, w, cps, labels, ts, **seg_kwargs):
    seg_df = seg_kwargs["segmentation"]
    found_cps = seg_df.loc[seg_df["dataset"] == dataset].iloc[0].found_cps

    if "classifier" in seg_kwargs:
        clf = seg_kwargs["classifier"]
    else:
        clf = "rocket"

    if "merge_score" in seg_kwargs:
        merge_score = seg_kwargs["merge_score"]
    else:
        merge_score = "cgain"

    if "window_size" in seg_kwargs:
        window_size = seg_kwargs["window_size"]
    else:
        window_size = "suss"

    clap = CLaP(window_size=window_size, classifier=clf, merge_score=merge_score, n_jobs=4)
    clap.fit(ts, found_cps)

    found_cps = clap.get_change_points()
    found_labels = clap.get_segment_labels()

    true_seg_labels = create_state_labels(cps, labels, ts.shape[0])
    pred_seg_labels = create_state_labels(found_cps, found_labels, ts.shape[0])

    return evaluate_state_detection_algorithm(dataset, ts.shape[0], cps, found_cps, true_seg_labels, pred_seg_labels,
                                              None, verbose=1)


# Runs window size ablation experiment
def evaluate_window_size(dataset_name, exp_path, n_jobs, verbose):
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    wss = ("mwf", "suss", "fft", "acf")

    seg_algo = "ClaSP"
    converters = dict([(column, lambda data: np.array(eval(data))) for column in ["found_cps"]])
    seg_df = pd.read_csv(
        f"../experiments/segmentation/{dataset_name}_{seg_algo}.csv.gz",
        converters=converters
    )[["dataset", "found_cps"]]

    for candidate_name in wss:
        print(f"Evaluating competitor: {candidate_name}")

        df = evaluate_candidate(
            dataset_name,
            candidate_name,
            eval_func=evaluate_clap,
            columns=None,
            n_jobs=n_jobs,
            verbose=verbose,
            segmentation=seg_df,
            window_size=candidate_name,
        )

        df.to_csv(f"{exp_path}{dataset_name}_{candidate_name}.csv.gz", compression='gzip')


# Runs segmentation algorithm ablation experiment
def evaluate_segmentation_algorithm(dataset_name, exp_path, n_jobs, verbose):
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    seg_algos = ("DummySeg", "ClaSP", "BinSeg", "Window", "Pelt", "FLUSS", "RuLSIF")  #

    converters = dict([(column, lambda data: np.array(eval(data))) for column in ["found_cps"]])

    for candidate_name in seg_algos:
        print(f"Evaluating competitor: {candidate_name}")

        seg_df = pd.read_csv(
            f"../experiments/segmentation/{dataset_name}_{candidate_name}.csv.gz",
            converters=converters
        )[["dataset", "found_cps"]]

        df = evaluate_candidate(
            dataset_name,
            candidate_name,
            eval_func=evaluate_clap,
            columns=None,
            n_jobs=n_jobs,
            verbose=verbose,
            segmentation=seg_df
        )

        df.to_csv(f"{exp_path}{dataset_name}_{candidate_name}.csv.gz", compression='gzip')


# Runs classification algorithm ablation experiment
def evaluate_classifier(dataset_name, exp_path, n_jobs, verbose):
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    classifiers = ("dummy", "rocket", "mrhydra", "quant", "weasel", "rdst", "freshprince", "proximityforest")

    seg_algo = "ClaSP"
    converters = dict([(column, lambda data: np.array(eval(data))) for column in ["found_cps"]])
    seg_df = pd.read_csv(
        f"../experiments/segmentation/{dataset_name}_{seg_algo}.csv.gz",
        converters=converters
    )[["dataset", "found_cps"]]

    for candidate_name in classifiers:
        print(f"Evaluating competitor: {candidate_name}")

        df = evaluate_candidate(
            dataset_name,
            candidate_name,
            eval_func=evaluate_clap,
            columns=None,
            n_jobs=n_jobs,
            verbose=verbose,
            segmentation=seg_df,
            classifier=candidate_name
        )

        df.to_csv(f"{exp_path}{dataset_name}_{candidate_name}.csv.gz", compression='gzip')


# Runs merge score ablation experiment
def evaluate_merge_score(dataset_name, exp_path, n_jobs, verbose):
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    merge_scores = ("cgain", "f1_score", "log_loss", "ami", "hamming_loss", "roc_auc")

    seg_algo = "ClaSP"
    converters = dict([(column, lambda data: np.array(eval(data))) for column in ["found_cps"]])
    seg_df = pd.read_csv(
        f"../experiments/segmentation/{dataset_name}_{seg_algo}.csv.gz",
        converters=converters
    )[["dataset", "found_cps"]]

    for candidate_name in merge_scores:
        print(f"Evaluating competitor: {candidate_name}")

        df = evaluate_candidate(
            dataset_name,
            candidate_name,
            eval_func=evaluate_clap,
            columns=None,
            n_jobs=n_jobs,
            verbose=verbose,
            segmentation=seg_df,
            merge_score=candidate_name,
        )

        df.to_csv(f"{exp_path}{dataset_name}_{candidate_name}.csv.gz", compression='gzip')


if __name__ == '__main__':
    exp_path = "../experiments/ablation_study/"
    n_jobs, verbose = 1, 0

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    evaluate_window_size("train", exp_path, n_jobs, verbose)
    evaluate_segmentation_algorithm("train", exp_path, n_jobs, verbose)
    evaluate_classifier("train", exp_path, n_jobs, verbose)
    evaluate_merge_score("train", exp_path, n_jobs, verbose)
