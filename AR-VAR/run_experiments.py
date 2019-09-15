# Ben Ma
# Python 3.x

from utils import load_data, split_data
from baseline import run_baseline
from autoregression import run_ar
from lasso import run_lasso
from ridge import run_ridge
import numpy as np
import csv
import pickle
import argparse

OUTPUT_BASENAME = "./results/new_compress"
ATTENTION_LEN = 40
DATASETS = ["happy_enjoyment.csv", "sad_long_enjoyment.csv", "sad_short_enjoyment.csv"]
HORIZONS = ["40", "80"]
CV_FOLDS = ["Beg", "End", "Avg"]
SCRIPTS = ["baseline", "ar", "lasso", "ridge"]
W_ENJOY_OR_NOT = ["w/enjoy", "audio only"]
NUM_FEATURES = 75
# for FEATURE_GROUPS, include all columns you want besides 0 (0 being the previous enjoyment values)

FEATURE_GROUPS = {
    "All" : [i for i in range(1, NUM_FEATURES)],
    "MFCCs" : [i for i in range(1, 1 + 3*13)],
    "Spectrogram": [41, 44, 45, 46, 47, 62, 63],
    "Harmony": [i for i in range(48, 60)] + [42, 60],
    "Dynamics": [1, 43, 61],
    "Rhythm": [40],
    "LPCs" : [i for i in range(64, 75)]
}

FEATURE_GROUPS = {
    "All" : [i for i in range(1, NUM_FEATURES)],
    "Dynamics (new compress)" : [1, 43, 61]
}

Dynamics = [1, 40, 43, 61]
Chroma = [i for i in range(48, 60)]
Spectrogram = [44, 45, 46, 47, 62, 63]

# FEATURE_GROUPS = {
#     "Dynamics": [1, 43, 61],
#     "Rhythm": [40]
# }

def main():
    # receive data_set
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', type=str, default='none')
    para = parser.parse_args()
    assert para.data_set != "none", "You must specify a data_set in the arguments! (Should be a path to a .csv file)"
    dataset = para.data_set

    print("Dataset "+dataset+"...")
    (X_full, y) = load_data(dataset)
    results = {}
    for feature_group in FEATURE_GROUPS.keys():
        results[feature_group] = {}
        for horizon in HORIZONS:
            results[feature_group][horizon] = {}
            for script in SCRIPTS:
                results[feature_group][horizon][script] = {}
                for enjoy_or_not in W_ENJOY_OR_NOT:
                    results[feature_group][horizon][script][enjoy_or_not] = {}
                    for cv_fold in CV_FOLDS:
                        results[feature_group][horizon][script][enjoy_or_not][cv_fold] = {}
    # results will be an x-level dict with the following arguments
    # results[feature_group][horizon][script][w_enjoy_or_not][cv_fold][RMSE_or_alpha]
    # e.g. results["MFCCs"]["40"][baseline]["audio only"]["Beg"]["RMSE"]

    for feature_group in FEATURE_GROUPS.keys():
        print("Feature Group "+feature_group+"...")
        # add selected columns in feature_group to X
        num_timesteps = X_full.shape[0]
        X_group = X_full[:, 0] # col 0 included by default
        try:
            FEATURE_GROUPS[feature_group].remove(0)
        except ValueError:
            pass
        for col in FEATURE_GROUPS[feature_group]: # include 0 and every column in feature groups
            X_group = np.hstack( (
                np.reshape(X_group, (num_timesteps, -1)),
                np.reshape(X_full[:, col], (num_timesteps, 1))
            ) )
        for script in SCRIPTS:
            for horizon in HORIZONS:
                X1 = X_group # X starts as the full X_group (will remove columns if necessary)
                if(script=="baseline" or script=="ar"):
                    X2 = np.reshape(X1[:, 0], (num_timesteps, 1)) # X is only first column
                else:
                    X2 = X1
                for enjoy_or_not in W_ENJOY_OR_NOT:
                    if (enjoy_or_not=="audio only" and (script=="lasso" or script=="ridge")):
                        X3 = X2[:, 1:] # exclude column 0
                    else:
                        X3 = X2
                    for cv_fold in CV_FOLDS:
                        # RUN EXPERIMENT AS LONG AS IT'S NOT "AUDIO ONLY" FOR BASELINE OR AR
                        # OR IF CV_FOLD IS AVG
                        if( (script=="baseline" or script=="ar") and (enjoy_or_not=="audio only") ):
                            results[feature_group][horizon][script][enjoy_or_not][cv_fold]["RMSE"] = \
                                results[feature_group][horizon][script]["w/enjoy"][cv_fold]["RMSE"]
                        elif( cv_fold=="Avg" ):
                            results[feature_group][horizon][script][enjoy_or_not][cv_fold]["RMSE"] = \
                                (results[feature_group][horizon][script][enjoy_or_not]["Beg"]["RMSE"] + \
                                results[feature_group][horizon][script][enjoy_or_not]["End"]["RMSE"]) / 2
                        else:
                            (X_train, y_train, X_test, y_test) = split_data(
                                X3, y, int(horizon), ATTENTION_LEN, cv_fold)
                            RMSE = -1
                            if script == "baseline":
                                RMSE = run_baseline( X_train, y_train, X_test, y_test )
                            elif script == "ar":
                                RMSE = run_ar(X_train, y_train, X_test, y_test)
                            elif script == "lasso":
                                (RMSE, Alpha, Coefs) = run_lasso(X_train, y_train, X_test, y_test)
                                results[feature_group][horizon][script][enjoy_or_not][cv_fold]["Alpha"] = Alpha
                                results[feature_group][horizon][script][enjoy_or_not][cv_fold]["Coefs"] = Coefs
                            elif script == "ridge":
                                (RMSE, Alpha, Coefs) = run_ridge(X_train, y_train, X_test, y_test)
                                results[feature_group][horizon][script][enjoy_or_not][cv_fold]["Alpha"] = Alpha
                                results[feature_group][horizon][script][enjoy_or_not][cv_fold]["Coefs"] = Coefs
                            results[feature_group][horizon][script][enjoy_or_not][cv_fold]["RMSE"] = RMSE
    # WRITE TO RESULTS TO FILE
    output_fname = OUTPUT_BASENAME+"_"+dataset
    pickle_fname = output_fname[:-3]+"p"
    # Write Pickle --------------------
    with open(pickle_fname, "wb") as f:
        pickle.dump(results, f)
    # Write CSV --------------------
    with open(output_fname, "w", newline='') as f:
        col_names = ["", "Beg w/enjoy", "End w/enjoy", "Avg w/enjoy", "Beg audio only", "End audio only", "Avg audio only"]
        row_names = ["baseline", "ar", "lasso", "lasso Alpha", "ridge", "ridge Alpha"]
        writer = csv.writer(f)
        for horizon in HORIZONS:
            for feature_group in FEATURE_GROUPS.keys():
                    writer.writerow([feature_group+" - Horizon "+horizon])
                    writer.writerow(col_names)
                    for row in row_names:
                        my_row = [row]
                        for enjoy_or_not in W_ENJOY_OR_NOT:
                            for cv_fold in CV_FOLDS:
                                if (row.find("Alpha") != -1): # alpha case
                                    if (cv_fold != "Avg"):
                                        my_row.append(
                                            results[feature_group][horizon][row[0:5]][enjoy_or_not][cv_fold]["Alpha"]
                                        )
                                    else:
                                        my_row.append("")
                                else: # RMSE case
                                    my_row.append(
                                        results[feature_group][horizon][row][enjoy_or_not][cv_fold]["RMSE"]
                                    )
                        writer.writerow(my_row)
                    writer.writerow([]) # put an empty line here
    print("Results written to "+output_fname+"\n")

if (__name__=="__main__"):
    main()