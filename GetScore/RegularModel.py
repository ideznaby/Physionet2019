# -*- coding: utf-8 -*-
import joblib
import sys
import numpy as np

#This type of model gets the input data and writes the output directly to the indicated output file
def Runregular(indata, covs, outfilepath, addprob = False, interpolate = True):
    outfile = open(outfilepath, 'w')
    clf = joblib.load('RF_model.pkl')
    X_test = []
    for row in indata:
        line_orig = row + covs
        row_final = line_orig
        X_test.append(row_final)
    probs = clf.predict_proba(X_test)
    print('PredictedProbability|PredictedLabel', file=outfile)
    #treshhold = 0.5
    seenone = False
    prob1 = 0
    index1 = 0
    for (i,P) in enumerate(probs):
        additionalprob = ((0.4 * i) / len(probs)) - 0.3
        if addprob:
            P[1] = P[1] + additionalprob
            if P[1] < 0:
                P[1] = 0
            elif P[1] > 1:
                P[1] = 1
        treshhold = 0.5
        label = int(P[1]>treshhold)
        if label == 1 and not seenone:
            seenone = True
            prob1 = P[1]
            index1 = i
        if interpolate:
            if seenone == False:
                print(str(P[1]) + '|' + str(label), file=outfile)
            else:
                prob = (((1 - prob1) / (len(probs)-index1)) * (i - index1)) + prob1
                print(str(prob) + '|' + str(1), file=outfile)
        else:
            print(str(P[1]) + '|' + str(label), file=outfile)
    outfile.close()