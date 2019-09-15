#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:17:45 2019

@author: greert
"""

from sklearn import linear_model
import pandas as pd
from heapq import nlargest
from sklearn import preprocessing
import math
import numpy as np
import matplotlib.pyplot as plt

from utils import params_setup, load_data

ALPHAS = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1]

# stores the result from a single alpha
class Result:
    def __init__(self,
             alpha,
             RMSE,
             coefs,
             coef_indices,
             bias):
        self.alpha = alpha
        self.RMSE = RMSE
        self.coefs = coefs
        self.coef_indices = coef_indices
        self.bias = bias

def run_lasso(X_train, y_train, X_test, y_test):
    results = []
    for aa in ALPHAS:
        clf = linear_model.Lasso(alpha=aa, normalize=False)
        clf.fit(X_train, y_train)

        # Determine which variables (timesteps, since this is AR) are most important

        # Get RMSE
        RMSE = ((len(y_test) ** -1) * sum((clf.predict(X_test) - y_test) ** 2))**0.5

        results.append(Result(aa,
                              RMSE,
                              clf.coef_,
                              [],
                              clf.intercept_))
    min_rmse = results[0].RMSE
    min_idx = 0
    for i in range(0, len(results)):
        if(results[i].RMSE<min_rmse):
            min_rmse = results[i].RMSE
            min_idx = i

    # Second round alpha-finding
    alphas2 = [ results[min_idx].alpha/10*3,
                results[min_idx].alpha/10*6,
                results[min_idx].alpha * 10 * 3,
                results[min_idx].alpha * 10 * 6,
                ]
    results2 = [results[min_idx]]
    for aa in alphas2:
        clf = linear_model.Lasso(alpha=aa, normalize=True)
        clf.fit(X_train, y_train)

        # Get RMSE
        RMSE = ((len(y_test) ** -1) * sum((clf.predict(X_test) - y_test) ** 2))**0.5

        results2.append(Result(aa,
                              RMSE,
                              clf.coef_,
                              [],
                              clf.intercept_))
    min_rmse = results2[0].RMSE
    min_idx = 0
    for i in range(0, len(results2)):
        if (results2[i].RMSE < min_rmse):
            min_rmse = results2[i].RMSE
            min_idx = i

    return (results2[min_idx].RMSE, results2[min_idx].alpha, results2[min_idx].coefs)

def main():
    #---------------LOAD PARAMETERS, INITIALIZE VARS---------------
    results = []
    para = params_setup("lasso")  # "lasso" for lasso VAR

    # Which variables are > thresh?
    thresh = para.variable_threshold
    # How far in the future are we predicting?
    offset = para.horizon
    # What is the window of previous timesteps we're looking at?
    attn_length = para.attention_len

    #---------------PREPARE DATA------------------
    (X_cut, y_cut) = load_data(para)

    num_timesteps = X_cut.shape[0]
    num_features = X_cut.shape[1]

    X = list()
    for i in range(attn_length, num_timesteps - offset):
        X.append((X_cut[i - attn_length : i, :]).flatten()) # must flatten (timesteps, features) into 1D b/c model only takes up to 2D
    X = np.array(X)
    y = y_cut[attn_length+offset:num_timesteps]

    # Split into training and testing
    if (para.test_set_at_beginning):
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

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    for aa in ALPHAS:
        clf = linear_model.Lasso(alpha=aa, normalize=True)
        clf.fit(X_train, y_train)

        #0 rated emotion signal
        #1-13 MFCCs
        #14-26 dMFCCs
        #27-39 ddMFCCs
        #40 clarity
        #41 brightness
        #42 key_strength
        #43 rms
        #44 centroid
        #45 spread
        #46 skewness
        #47 kurtosis
        #48-59 chroma
        #60 mode
        #61 compress
        #62 hcdf
        #63 flux
        #64-74 lpcs

        # Determine which variables (timesteps, since this is AR) are most important

        inds = [j for (i, j) in zip(clf.coef_, range(len(clf.coef_))) if abs(i) >= thresh]
        inds_mod = [x % num_features for x in inds]

        # Get RMSE
        RMSE = ((len(y_test) ** -1) * sum((clf.predict(X_test) - y_test) ** 2))**0.5
        print("RMSE for alpha "+str(aa)+": " + (str)(RMSE))
        if (para.show_plots):
            plt.plot(range(len(y_test)),clf.predict(X_test))
            plt.plot(range(len(y_test)),y_test)
            plt.title('Lasso predictions vs truth (alpha='+str(aa)+")")
            plt.legend(['predictions', 'truth'], loc='upper right')
            plt.show()

        results.append(Result(aa,
                              RMSE,
                              clf.coef_,
                              sorted(inds_mod),
                              clf.intercept_))
    min_rmse = results[0].RMSE
    min_idx = 0
    for i in range(0, len(results)):
        if(results[i].RMSE<min_rmse):
            min_rmse = results[i].RMSE
            min_idx = i
    print("Minimum RMSE: "+str(min_rmse)+" for alpha="+str(results[min_idx].alpha))
    # ----------------- WRITE RESULT OF BEST ALPHA TO FILE -------------------
    best_result = results[min_idx]
    with open(para.output_filename, 'w') as f:
        f.write("RMSE: "+(str)(best_result.RMSE))
        f.write("\nAlpha: "+(str)(best_result.alpha))
        f.write("\nCoefficients:\n")
        f.write(np.array2string(best_result.coefs, threshold=np.nan))
        f.write("\nCoefficient indices over threshold "+str(para.variable_threshold)+":\n")
        f.write(' '.join(str(x) for x in best_result.coef_indices))
        f.write("\nTotal number of coefficients over threshold: "+str(len(best_result.coef_indices)))
        f.write("\nRegression bias term: "+str(best_result.bias))
        # Write overall RMSE for each alpha as well
        f.write("\n\nRMSEs for each alpha:\n")
        for result in results:
            f.write("Alpha: "+str(result.alpha)+",\tRMSE: "+str(result.RMSE)+"\n")
        print("Successfully wrote results to file "+para.output_filename)

if __name__ == '__main__':
    main()