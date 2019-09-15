#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:12:28 2019

@author: greert
"""

# BARIMA (Baseline ARIMA) is essentially a "dumb" AR with p = 1, d = 0, and q = 0

from statsmodels.tsa.arima_model import ARIMA
import numpy as np
import math
from utils import params_setup, load_data
import matplotlib.pyplot as plt
import copy
import argparse
import pickle
import csv

OUTPUT_BASENAME = "./results/test"
ATTENTION_LEN = 1
HORIZONS = ["40", "80"]
CV_FOLDS = ["Beg", "End"]

def run_barima(X_train, y_train, X_test, y_test):
    model = ARIMA(history, order=(1, 0, 0))
    model_fit = model.fit(disp=0)
    RMSE = ((len(y_test) ** -1) * sum((clf.predict(X_test) - y_test) ** 2)) ** 0.5
    return RMSE

def main():
    # LOAD IN DATA
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', type=str, default='none')
    para = parser.parse_args()
    assert para.data_set != "none", "You must specify a data_set in the arguments! (Should be a path to a .csv file)"
    dataset = para.data_set

    print("Dataset " + dataset + "...")
    (_, y) = load_data(dataset)

    # store results in this dict
    results = {}
    for horizon in HORIZONS:
        results[horizon] = {}
        for cv in CV_FOLDS:
            results[horizon][cv] = {}
    for cv in CV_FOLDS:
        # SPLIT INTO TRAINING AND TEST
        cutoff = int(len(y) * 0.8)
        train, test = y[0:cutoff], y[cutoff:len(y)]

        # TRAIN MODEL
        X = copy.deepcopy(train)
        model = ARIMA(X, order=(ATTENTION_LEN, 0, 0))
        model_fit = model.fit()

        # GET PREDICTION ERROR FOR EACH HORIZON
        for horizon in HORIZONS:
            h = int(horizon)
            y_hat = model_fit.predict(start=X.shape[0] + h - 1, end=X.shape[0] + test.shape[0] - 1)[h:]
            predictions = y_hat
            truth = test[h:]
            RMSE = ((len(truth) ** -1) * sum((predictions - truth) ** 2)) ** 0.5
            results[horizon][cv]["RMSE"] = RMSE

            # save test set predictions and truths
            results[horizon][cv]["test_predictions"] = predictions
            results[horizon][cv]["test_truth"] = truth

            # save training set predictions and truth
            if(cv=="Beg"):
                y_hat_train = model_fit.predict(start= h, end=train.shape[0]-1)
                predictions_train = y_hat_train
                truth_train = train[h:]
                results[horizon][cv]["train_predictions"] = predictions_train
                results[horizon][cv]["train_truth"] = truth_train
                RMSE_train = ((len(truth_train) ** -1) * sum((predictions_train - truth_train) ** 2)) ** 0.5
                print("Train RMSE:", RMSE_train)
            print("Test RMSE:", RMSE)


    # CALCULATE AVG RESULT FOR EACH HORIZON
    for horizon in HORIZONS:
        results[horizon]["Avg"] = {}
        results[horizon]["Avg"]["RMSE"] = (results[horizon]["Beg"]["RMSE"] + results[horizon]["End"]["RMSE"])/2

    # OUTPUT RESULTS TO FILE
    dataset_name = dataset.replace(".csv", "")
    file_out = OUTPUT_BASENAME+"_"+dataset_name+".txt"
    pickle_out = OUTPUT_BASENAME + "_" + dataset_name + ".p"
    graphs_out = OUTPUT_BASENAME +"_" + dataset_name + "_*_graphs_h_?.csv"
    model_out = OUTPUT_BASENAME + "_" + 'ARIMA_weights_'+dataset_name+'.p'
    # OUTPUT RMSE FILE AND GRAPH FILES
    with open(file_out, 'w') as f:
        for horizon in HORIZONS:
            f.write("Horizon "+horizon+"\n")
            for cv in results[horizon].keys():
                RMSE = results[horizon][cv]["RMSE"]
                val = round(RMSE*100, 3) # round to three decimals
                f.write("RMSE ("+cv+"): "+str(val)+"\n")
            f.write("\n")
            # Output Graph Files
            for set in ["train", "test"]:
                with open(graphs_out.replace("*", set).replace("?", horizon), "w", newline="") as g:
                    writer = csv.writer(g)
                    truth = results[horizon]["Beg"][set+"_truth"]
                    predictions = results[horizon]["Beg"][set+"_predictions"]
                    assert truth.shape==predictions.shape, "Error: Truth and Predictions are not same shape!"
                    for i in range(truth.shape[0]):
                        writer.writerow([predictions[i], truth[i]])

    # OUTPUT PICKLE FILE
    with open(pickle_out, "wb") as f:
        pickle.dump(results, f)

    # OUTPUT ARIMA MODEL
    def __getnewargs__(self):
        return ((self.endog), (self.k_lags, self.k_diff, self.k_ma))
    ARIMA.__getnewargs__ = __getnewargs__

    model_fit.save(model_out)

if __name__ == '__main__':
    main()