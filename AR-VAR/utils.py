import argparse
import numpy as np
import csv
import math

# model_type is a string indicating what prefix should be added to the default filename
def params_setup(model_type):
    parser = argparse.ArgumentParser()
    parser.add_argument('--attention_len', type=int, default=40)
    parser.add_argument('--data_set', type=str, default='none')
    parser.add_argument('--horizon', type=int, default=40)
    parser.add_argument('--output_filename', type=str, default="none")
    parser.add_argument('--variable_threshold', type=float, default=0.5)
    parser.add_argument('--time_snip', type=int, default=0) # seconds to snip from beginning
    parser.add_argument('--sample_rate', type=int, default=40) # sample rate in Hz
    parser.add_argument('--show_plots', action='store_true') # whether to show plots or not
    parser.add_argument('--test_set_at_beginning', action='store_true') # whether to use test set as first 20% rather than last 20%

    para = parser.parse_args()

    assert para.data_set is not "none", "You must specify a data_set in the arguments! (Should be a path to a .csv file)"
    if para.output_filename == "none":
        para.output_filename = "./results/"+model_type+"_"+para.data_set+"_al_"+(str)(para.attention_len)+"_h_"+(str)(para.horizon)+".txt"
        print("Warning: output_filename not specified in arguments. Defaulting to "+para.output_filename)

    return para

def load_data(data_set):
    # Load X
    with open(data_set, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)

    X = np.array(your_list, dtype=np.float)
    print("Loaded array size: ", X.shape)

    # Load y (y will be first column in X)
    y = X[:, 0]

    # Snip out first 200 and last 400 samples (weird sampling goes on at end)
    X_cut = X[200:-400]
    y_cut = y[200:-400]

    X_norm = X_cut / X_cut.max(axis=0)
    y_norm = y_cut / y_cut.max(axis=0)
    X_norm_ptp = X_cut / X_cut.max(axis=0)
    y_norm_ptp = (y_cut - y_cut.min(0)) / y_cut.ptp(0)

    return (X_norm, y_norm) # (X_cut, y_cut)

# splits data appropriately into (X_train, y_train, X_test, y_test)
def split_data(X_orig, y_orig, horizon, attn_length, cv_fold):
    num_timesteps = X_orig.shape[0]

    X_list = list()
    for i in range(attn_length, num_timesteps - horizon):
        X_list.append((X_orig[i - attn_length : i, :]).flatten()) # must flatten (timesteps, features) into 1D b/c model only takes up to 2D
    X = np.array(X_list)
    y = y_orig[attn_length+horizon:num_timesteps]

    if (cv_fold is "Beg"):
        cutoff_idx = math.floor(X.shape[0] * 0.2)
        X_test = X[0: cutoff_idx, :]
        y_test = y[0: cutoff_idx]

        X_train = X[cutoff_idx:, :]
        y_train = y[cutoff_idx:]
    else:
        cutoff_idx = math.floor(X.shape[0] * 0.8)
        X_train = X[0: cutoff_idx, :]
        y_train = y[0: cutoff_idx]

        X_test = X[cutoff_idx:, :]
        y_test = y[cutoff_idx:]

    return (X_train, y_train, X_test, y_test)