import logging
import sys
sys.path.insert(0, "../")

from benchmark.state_detection_test import evaluate_state_detection_algorithm, evaluate_clap
from src.clap import CLaP

import pandas as pd

from src.utils import create_state_labels, load_has_datasets, load_tssb_datasets, load_datasets

import numpy as np

np.random.seed(1379)

# Computes an example of CLaP
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    benchmark = "TSSB"

    if benchmark == "TSSB":
        df_data = load_tssb_datasets()
    elif benchmark == "UTSA":
        df_data = load_datasets("UTSA")
    elif benchmark == "HAS":
        df_data = load_has_datasets()
    else:
        df_data = load_datasets(benchmark)

    idx = 15 # Crop example
    dataset, w, cps, labels, ts = df_data.iloc[idx, :]

    # Load segmentation for CLaP
    segmentation_algorithm = "ClaSP"
    converters = dict([(column, lambda data: np.array(eval(data))) for column in ["found_cps"]])
    seg_df = pd.read_csv(
        f"../experiments/segmentation/{benchmark}_{segmentation_algorithm}.csv.gz",
        converters=converters
    )[["dataset", "found_cps"]]

    found_cps = seg_df.loc[seg_df["dataset"] == dataset].iloc[0].found_cps
    evaluate_clap(dataset, w, cps, labels, ts, segmentations={"ClaSP": seg_df})

    clap = CLaP(n_jobs=4)
    clap.fit(ts, found_cps)

    found_cps = clap.get_change_points()
    found_labels = clap.get_segment_labels()

    true_seg_labels = create_state_labels(cps, labels, ts.shape[0])
    pred_seg_labels = create_state_labels(found_cps, found_labels, ts.shape[0])

    evaluate_state_detection_algorithm(dataset, ts.shape[0], cps, found_cps, true_seg_labels, pred_seg_labels, 0,
                                       verbose=1)
