# Ben Ma
# Python 3
# Extracts the RMS value out of each column of each dataset file.
# We will use these RMS values to multiply by the coefficient of each column to get an accurate "weight"
# on the model's final prediction.

import pickle
import numpy as np
from utils import load_data

DATASETS = ["happy_enjoyment.csv", "sad_long_enjoyment.csv", "sad_short_enjoyment.csv"]
OUTPUT_BASENAME = "./results/RMS_"

for dataset in DATASETS:
    # open CSV, read in columns into numpy array
    (X, _) = load_data(dataset)

    RMS = []
    # for each column...
    for i in range(X.shape[1]):
        col = X[:, i]
        # determine RMS value
        rms = np.sqrt(np.mean(col ** 2))
        # add RMS value to RMS array
        print("col", i, "RMS:", rms)
        RMS.append(rms)

    # pickle the RMS array
    output_name = OUTPUT_BASENAME+dataset.replace(".csv", ".p")
    with open(output_name, "wb") as f:
        pickle.dump(RMS, f)