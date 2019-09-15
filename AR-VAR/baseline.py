#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:12:28 2019

@author: greert
"""

from sklearn import linear_model
import pandas as pd
from sklearn import preprocessing
import numpy as np
import math
from utils import params_setup, load_data
import matplotlib.pyplot as plt

def run_baseline(X_train, y_train, X_test, y_test):
    clf = linear_model.LinearRegression(normalize=False)
    (X_train, X_test) = (np.reshape(X_train[:, -1], (-1, 1)),
                         np.reshape(X_test[:, -1], (-1, 1))
                         ) # for baseline, only get latest timestep
    clf.fit(X_train, y_train)
    RMSE = ((len(y_test) ** -1) * sum((clf.predict(X_test) - y_test) ** 2)) ** 0.5
    return RMSE

def main():
    para = params_setup("baseline")
    clf = linear_model.LinearRegression(normalize=True)

    #Which variables are > thresh?
    thresh = para.variable_threshold
    #How far in the future are we predicting?
    offset = para.horizon
    #What is the length of the autoregression
    attn_length = 1 # FIXED AT 1 FOR BASELINE

    (X_cut, y_cut) = load_data(para)


    #Make the autoregressive IVs
    y_regress = list()
    for i in range(offset+attn_length,len(y_cut)):
        y_regress.append(y_cut[i-attn_length-offset:i-offset])
    y_regress = np.array(y_regress)

    #Split into training and testing
    if(para.test_set_at_beginning):
        cutoff_idx = math.floor(y_regress.shape[0] * 0.2)
        X_test = y_regress[0: (cutoff_idx - attn_length - offset), :]
        y_test = y_cut[attn_length + offset:cutoff_idx]

        X_train = y_regress[cutoff_idx - attn_length - offset:, :]
        y_train = y_cut[cutoff_idx:]
    else:
        cutoff_idx = math.floor (y_regress.shape[0] * 0.8)
        X_train = y_regress[0: (cutoff_idx-attn_length-offset), :]
        y_train = y_cut[attn_length+offset:cutoff_idx]

        X_test = y_regress[cutoff_idx-attn_length-offset:, :]
        y_test = y_cut[cutoff_idx:]

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    clf.fit(X_train, y_train)

    # 0 rated emotion signal
    # 1-13 MFCCs
    # 14-26 dMFCCs
    # 27-39 ddMFCCs
    # 40 clarity
    # 41 brightness
    # 42 key_strength
    # 43 rms
    # 44 centroid
    # 45 spread
    # 46 skewness
    # 47 kurtosis
    # 48-59 chroma
    # 60 mode
    # 61 compress
    # 62 hcdf
    # 63 flux
    # 64-74 lpcs


    #Determine which variables (timesteps, since this is AR) are most important

    inds = [j for (i,j) in zip(clf.coef_,range(len(clf.coef_))) if abs(i) >= thresh]

    # print(sorted(inds))
    # print(sum(abs(clf.coef_)>thresh))
    # print((abs(clf.coef_)>thresh))
    # print(nlargest(3, clf.coef_))
    # print(clf.intercept_)

    #Get RMSE
    RMSE = ((len(y_test)**-1)*sum((clf.predict(X_test)-y_test)**2)) ** 0.5
    print("RMSE: "+(str)(RMSE))
    if(para.show_plots):
        plt.plot(range(len(y_test[:-2])),clf.predict(X_test[:][:-2]))
        plt.plot(range(len(y_test[:-2])),y_test[:-2])
        plt.title('Autoregression predictions vs truth')
        plt.legend(['predictions', 'truth'], loc='upper right')
        plt.show()

    with open(para.output_filename, 'w') as f:
        f.write("RMSE: "+(str)(RMSE))
        f.write("\nCoefficients:\n")
        f.write(np.array2string(clf.coef_, threshold=np.nan))
        f.write("\nCoefficient indices over threshold "+str(para.variable_threshold)+":\n")
        f.write(' '.join(str(x) for x in sorted(inds)))
        f.write("\nTotal number of coefficients over threshold: "+str(sum(abs(clf.coef_)>thresh)))
        f.write("\nRegression bias term: "+str(clf.intercept_))
        print("Successfully wrote results to file "+para.output_filename)

if __name__ == '__main__':
    main()