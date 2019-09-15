import pickle
import copy
import numpy as np
import csv

SONGS = ["happy", "sad_short", "sad_long"]
FILE = "../ben_auditory_results/new_compress_*_enjoyment_new_compress.p"
RMS_FILE = "./results/RMS_*_enjoyment.p"
OUT_FILE = "./results/new_compress_relevant_features_*.csv"
HORIZONS = ["40", "80"]
CV_FOLDS = ["Beg", "End", "Avg"]
SCRIPTS = ["baseline", "ar", "lasso", "ridge"]
W_ENJOY_OR_NOT = ["w/enjoy", "audio only"]
ATTN_LEN = 40
NUM_FEATURES = 75
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

# multiplies each coef by its corresponding column RMS
def apply_RMS(raw_coefs, RMS, ft_group, w_enjoy):
    new_coefs = copy.deepcopy(raw_coefs)
    cols_included = copy.deepcopy(FEATURE_GROUPS[ft_group]) # cols_included is a list of ints (the indices of appropriate columns)
    if(w_enjoy=="w/enjoy"):
        cols_included.insert(0, 0) # insert col 0 first if enjoy col included
    assert new_coefs.shape[0]%len(cols_included)==0, \
        "Mismatch between number of coefficients ("+str(new_coefs.shape[0]) + \
        ") and number of RMS columns (" + str(len(cols_included)) +")"
    for i in range(len(cols_included)):
        col = cols_included[i]
        for j in range(new_coefs.shape[0] // len(cols_included)): # hit every timestep for this feature
            idx = i + len(cols_included)*j
            new_coefs[idx] *= RMS[col]
    return new_coefs

def main():
    for song in SONGS:
        result_file = FILE.replace("*", song)
        rms_file = RMS_FILE.replace("*", song)
        out_file = OUT_FILE.replace("*", song)
        with open(result_file, "rb") as f:
            results = pickle.load(f)
        with open(rms_file, "rb") as f:
            RMS = pickle.load(f)
        score_dict = {}
        for w_enjoy_or_not in W_ENJOY_OR_NOT:
            score_dict[w_enjoy_or_not] = {}
            for feature_group in FEATURE_GROUPS.keys():
                if(w_enjoy_or_not=="w/enjoy"):
                    scores = np.zeros((len(FEATURE_GROUPS[
                                               feature_group]) + 1))  # scores will be a NP array of values, corresponding to each feature column's total "score" (sum of weights)
                    scores_abs = np.zeros((len(FEATURE_GROUPS[feature_group]) + 1))
                else: # audio only
                    scores = np.zeros((len(FEATURE_GROUPS[
                                               feature_group])))  # scores will be a NP array of values, corresponding to each feature column's total "score" (sum of weights)
                    scores_abs = np.zeros((len(FEATURE_GROUPS[feature_group])))
                score_tuples = []
                for horizon in HORIZONS:
                    if(w_enjoy_or_not=="w/enjoy"):
                        script="ridge"
                    else:
                        script="lasso"
                    for cv_fold in ["Beg", "End"]:
                        raw_coefs = results[feature_group][horizon][script][w_enjoy_or_not][cv_fold]["Coefs"]
                        num_features = raw_coefs.shape[0] / ATTN_LEN
                        coefs = copy.deepcopy(raw_coefs) # apply_RMS(raw_coefs, RMS, feature_group, w_enjoy_or_not)
                        coefs_reshape = np.reshape(coefs, (-1, ATTN_LEN), order='F')
                        for feature in range(coefs_reshape.shape[0]):
                            scores[feature] += np.sum(coefs_reshape[feature])
                            scores_abs[feature] += np.sum(np.abs(coefs_reshape[feature]))
                # create score tuples from scores and scores_abs
                for i in range(0, scores.shape[0]): # let's include 0, the previous affective signal
                    if(w_enjoy_or_not=="w/enjoy"):
                        if(i==0):
                            absolute_column = 0
                        else:
                            absolute_column = FEATURE_GROUPS[feature_group][i-1]
                    else: # audio only
                        absolute_column = FEATURE_GROUPS[feature_group][i]
                    score_tuples.append( (absolute_column, scores_abs[i], scores[i]) )
                # sort tuples
                scores_sorted = sorted(score_tuples, key=lambda x: x[1], reverse=True)
                score_dict[w_enjoy_or_not][feature_group] = scores_sorted

        # WRITE TO RESULTS TO FILE
        with open(out_file, "w", newline='') as f:
            writer = csv.writer(f)
            for w_enjoy_or_not in W_ENJOY_OR_NOT:
                for feature_group in FEATURE_GROUPS.keys():
                    writer.writerow([feature_group+" - "+w_enjoy_or_not])
                    writer.writerow(["Feature", "Score (Absolute)", "Score"])
                    for tup in score_dict[w_enjoy_or_not][feature_group]:
                        writer.writerow(tup)
                    writer.writerow([])

if __name__=="__main__":
    main()