import os

ABS_PATH = os.path.dirname(os.path.abspath(__file__))

import tempfile

import numpy as np


# To run this method:
#   1. download the adapted autoplait from https://sites.google.com/site/onlinesemanticsegmentation/
#   2. place it into c/autoplait
#   3. build it (make clean autoplait)
def autoplait(name, ts, n_cps):
    # Raw name sometimes leads to errors with filesystem handling
    name = f"{hash(name)}"

    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = os.path.join(temp_dir, f"AutoPlait-{name}/")
        autoplait_path = os.path.join(ABS_PATH, "../../c/autoplait/")

        if not os.path.exists(tmp_path):
            os.system(f"mkdir {tmp_path}")

        np.savetxt(os.path.join(tmp_path, f"{name}.txt"), ts)
        np.savetxt(os.path.join(tmp_path, "list"), [os.path.join(tmp_path, f"{name}.txt")], fmt="%s")

        os.system(
            f"{os.path.join(autoplait_path, 'autoplait')} 1 {n_cps + 1} {os.path.join(tmp_path, 'list')} {tmp_path}")  #

        segment_ids = []
        found_cps = []

        segment_i = 0

        while os.path.exists(os.path.join(tmp_path, f"segment.{segment_i}")):
            pred = np.loadtxt(os.path.join(tmp_path, f"segment.{segment_i}"), dtype=np.float64)

            if len(pred.shape) == 1:
                pred = [pred[1]]
            elif len(pred.shape) == 2:
                pred = pred[:, 1]
            else:
                assert False

            found_cps.extend(pred)
            segment_ids.append(segment_i)
            segment_i += 1

        found_labels = []

        if os.path.exists(os.path.join(tmp_path, f"segment.labels")):
            with open(os.path.join(tmp_path, f"segment.labels"), 'r') as f:
                for line in f.readlines():
                    line = line.split("\t\t")

                    try:
                        found_labels.append(int(line[1]))
                    except:
                        found_labels.append(0)

    sorted_args = np.argsort(found_cps)

    # Fails sometimes
    if len(found_cps) != len(found_labels):
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    return np.array(found_cps)[sorted_args][:-1], np.array(found_labels)[sorted_args]
