# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 04:02:03 2019

@author: Iman
"""
import joblib
import sys
import numpy as np
if __name__ == '__main__':
    globalavg = np.array([ 84.58144338,  97.19395453,  36.97722824, 123.7504654 ,
        82.40009989,  63.83055577,  18.72649786,  32.95765667,
        -0.68991919,  24.07548056,   0.55483863,   7.37893403,
        41.02186881,  92.65418775, 260.22338482,  23.91545211,
       102.48366144,   7.55753085, 105.82790991,   1.51069935,
         1.83617726, 136.93228329,   2.64666602,   2.05145021,
         3.54423765,   4.13552797,   2.11405946,   8.29009945,
        30.79409334,  10.43083279,  41.23119346,  11.44640502,
       287.38570592, 196.01391079,  62.00946888,   0.55926904,
         0.49657112,   0.50342888, -56.12512177,  26.9949923 ])
    clf = joblib.load('RF_model.pkl')
    infile = open(sys.argv[1], 'r')
    next(infile)
    outfile = open(sys.argv[2], 'w')
    X_test = []
    feature_avg = np.zeros(40)
    feature_sum = np.zeros(40)
    feature_count = np.zeros(40)
    for row in infile:
        line_orig = row.split('|')
        row_final = np.zeros(40)
        for (j, val) in enumerate(line_orig):
            if val != 'NaN':
                feature_sum[j] += float(val)
                feature_count[j] += 1
        for (i, val) in enumerate(line_orig):
            if val == 'NaN':
                if feature_count[i] != 0:
                    row_final[i] = feature_avg[i] / feature_count[i]
                else:
                    row_final[i] = globalavg[i]
            else:
                row_final[i] = float(val)
        X_test.append(row_final)
    probs = clf.predict_proba(X_test)
    print('PredictedProbability|PredictedLabel', file=outfile)
    treshhold = 0.9
    seenone = False
    prob1 = 0
    index1 = 0
    for (i,P) in enumerate(probs):
        #treshhold = (-0.5 * i) / len(probs) + 0.9
        treshhold = 0.5
        label = int(P[1]>treshhold)
        if label == 1 and not seenone:
            seenone = True
            prob1 = P[1]
            index1 = i
        if seenone == False:
            print(str(P[1]) + '|' + str(label), file=outfile)
        else:
            prob = (((1 - prob1) / (len(probs)-index1)) * (i - index1)) + prob1
            print(str(prob) + '|' + str(1), file=outfile)
    outfile.close()