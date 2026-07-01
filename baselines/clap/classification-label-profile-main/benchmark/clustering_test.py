import os
import sys
sys.path.insert(0, "../")

from sklearn.cluster import AgglomerativeClustering, SpectralClustering

from external.competitor.time2feat import feature_extraction, feature_selection, ClusterWrapper

from aeon.clustering import TimeSeriesKMedoids, TimeSeriesKMeans
from tslearn.clustering import KShape, KernelKMeans
from sklearn.metrics import adjusted_mutual_info_score

import daproli as dp
import pandas as pd
from tqdm import tqdm

from src.utils import create_state_labels, load_tssb_datasets, create_sliding_window, expand_label_sequence, \
    load_has_datasets, load_datasets, load_train_dataset

import numpy as np

np.random.seed(1379)


# Runs k-Shape experiment
def evaluate_kshape(dataset, w, cps, labels, ts, **seg_kwargs):
    # as in CLaP
    sample_size = 2 * w
    stride = sample_size // 2

    clf = KShape(n_clusters=np.unique(labels).shape[0], random_state=1379)

    windows = create_sliding_window(ts, sample_size, stride)
    pred = clf.fit_predict(windows)

    pred_seg_labels = expand_label_sequence(pred, sample_size, stride)
    true_seg_labels = create_state_labels(cps, labels, pred_seg_labels.shape[0])

    return evaluate_clustering_detection_algorithm(dataset, true_seg_labels, pred_seg_labels)


# Runs k-Means experiment
def evaluate_kmeans(dataset, w, cps, labels, ts, **seg_kwargs):
    # as in CLaP
    sample_size = 2 * w
    stride = sample_size // 2

    clf = TimeSeriesKMeans(n_clusters=np.unique(labels).shape[0], random_state=1379)

    windows = create_sliding_window(ts, sample_size, stride)
    if ts.ndim > 1: windows = np.array([w.T for w in windows])

    try:
        pred = clf.fit_predict(windows)
    except ValueError as e:
        print(f"Exception: {e}; using only zero class.")
        pred = np.zeros(windows.shape[0], dtype=int)

    pred_seg_labels = expand_label_sequence(pred, sample_size, stride)
    true_seg_labels = create_state_labels(cps, labels, pred_seg_labels.shape[0])

    return evaluate_clustering_detection_algorithm(dataset, true_seg_labels, pred_seg_labels)


# Runs GAK experiment
def evaluate_gak(dataset, w, cps, labels, ts, **seg_kwargs):
    # as in CLaP
    sample_size = 2 * w
    stride = sample_size // 2

    clf = KernelKMeans(n_clusters=np.unique(labels).shape[0], random_state=1379)

    windows = create_sliding_window(ts, sample_size, stride)

    pred = clf.fit_predict(windows)
    if pred is None: pred = np.zeros(windows.shape[0], dtype=int)

    pred_seg_labels = expand_label_sequence(pred, sample_size, stride)
    true_seg_labels = create_state_labels(cps, labels, pred_seg_labels.shape[0])

    return evaluate_clustering_detection_algorithm(dataset, true_seg_labels, pred_seg_labels)


# Runs k-Medoids experiment
def evaluate_kmedoids(dataset, w, cps, labels, ts, **seg_kwargs):
    # as in CLaP
    sample_size = 2 * w
    stride = sample_size // 2

    clf = TimeSeriesKMedoids(n_clusters=np.unique(labels).shape[0], random_state=1379)

    windows = create_sliding_window(ts, sample_size, stride)
    if ts.ndim > 1: windows = np.array([w.T for w in windows])

    try:
        pred = clf.fit_predict(windows)
    except ValueError as e:
        print(f"Exception: {e}; using only zero class.")
        pred = np.zeros(windows.shape[0], dtype=int)

    pred_seg_labels = expand_label_sequence(pred, sample_size, stride)
    true_seg_labels = create_state_labels(cps, labels, pred_seg_labels.shape[0])

    return evaluate_clustering_detection_algorithm(dataset, true_seg_labels, pred_seg_labels)


# Runs Time2Feat experiment
def evaluate_time2feat(dataset, w, cps, labels, ts, **seg_kwargs):
    # as in CLaP
    sample_size = 2 * w
    stride = sample_size // 2

    windows = create_sliding_window(ts, sample_size, stride)
    if ts.ndim == 1: windows = np.array([np.array([w]) for w in windows])

    # model params
    transform_type = 'minmax'
    model_type = 'Hierarchical'
    context = {'model_type': model_type, 'transform_type': transform_type}

    try:
        df_features = feature_extraction(windows, batch_size=500)
        top_features = feature_selection(df_features, None, context)
        df_features = df_features[top_features]
        model = ClusterWrapper(n_clusters=np.unique(labels).shape[0], model_type=model_type,
                               transform_type=transform_type)
        pred = model.fit_predict(df_features.values)
    except Exception as e:
        print(f"Exception: {e}; using only zero class.")
        pred = np.zeros(shape=windows.shape[0], dtype=int)

    pred_seg_labels = expand_label_sequence(pred, sample_size, stride)
    true_seg_labels = create_state_labels(cps, labels, pred_seg_labels.shape[0])

    return evaluate_clustering_detection_algorithm(dataset, true_seg_labels, pred_seg_labels)


# Runs Agglomerative Clustering experiment
def evaluate_agglomerative(dataset, w, cps, labels, ts, **seg_kwargs):
    # as in CLaP
    sample_size = 2 * w
    stride = sample_size // 2

    clf = AgglomerativeClustering(n_clusters=np.unique(labels).shape[0])

    windows = create_sliding_window(ts, sample_size, stride)
    if ts.ndim > 1: windows = np.array([w.flatten() for w in windows])

    pred = clf.fit_predict(windows)

    pred_seg_labels = expand_label_sequence(pred, sample_size, stride)
    true_seg_labels = create_state_labels(cps, labels, pred_seg_labels.shape[0])

    return evaluate_clustering_detection_algorithm(dataset, true_seg_labels, pred_seg_labels)


# Runs Spectral Clustering experiment
def evaluate_spectral(dataset, w, cps, labels, ts, **seg_kwargs):
    # as in CLaP
    sample_size = 2 * w
    stride = sample_size // 2

    clf = SpectralClustering(n_clusters=np.unique(labels).shape[0], random_state=1379)

    windows = create_sliding_window(ts, sample_size, stride)
    if ts.ndim > 1: windows = np.array([w.flatten() for w in windows])

    pred = clf.fit_predict(windows)

    pred_seg_labels = expand_label_sequence(pred, sample_size, stride)
    true_seg_labels = create_state_labels(cps, labels, pred_seg_labels.shape[0])

    return evaluate_clustering_detection_algorithm(dataset, true_seg_labels, pred_seg_labels)


# Evaluates clustering experiment
def evaluate_clustering_detection_algorithm(dataset, labels_true, labels_pred):
    ami = np.round(adjusted_mutual_info_score(labels_true, labels_pred), 3)
    print(f"{dataset}: AMI-Score: {ami}")
    return dataset, labels_true.tolist(), labels_pred.tolist(), ami


# Runs an experiment for a given data set and algorithm
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
        columns = ["dataset", "true_labels", "found_labels", "ami_score"]

    df_cand = pd.DataFrame.from_records(
        df_cand,
        index="dataset",
        columns=columns,
    )

    print(f"{candidate_name}: mean_ami_score={np.round(df_cand.ami_score.mean(), 3)}")
    return df_cand


# Runs experiments for multiple algorithms and a single data set
def evaluate_competitors(dataset_name, exp_path, n_jobs, verbose):
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    competitors = [
        ("KShape", evaluate_kshape),
        ("KMeans", evaluate_kmeans),
        ("GAK", evaluate_gak),
        ("KMedoids", evaluate_kmedoids),
        ("Time2Feat", evaluate_time2feat),
        ("Agglomerative", evaluate_agglomerative),
        ("Spectral", evaluate_spectral)
    ]

    for candidate_name, eval_func in competitors:
        print(f"Evaluating competitor: {candidate_name}")

        df = evaluate_candidate(
            dataset_name,
            candidate_name,
            eval_func=eval_func,
            columns=None,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        df.to_csv(f"{exp_path}{dataset_name}_{candidate_name}.csv.gz", compression='gzip')


if __name__ == '__main__':
    exp_path = "../experiments/clustering/"
    n_jobs, verbose = 1, 0

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    for bench in ("TSSB", "UTSA", "HAS", "SKAB"):
        evaluate_competitors(bench, exp_path, n_jobs, verbose)
