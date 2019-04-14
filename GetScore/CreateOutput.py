# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:20:11 2019

@author: Iman
"""
import numpy as np
import os
import pickle
from PredRegModel import Run
from RegularModel import Runregular

def CreateOut(ID, oneprob, index, length, delta=5, outfolder = ''):
    zeroprob = 1 - oneprob
    probs = np.zeros(length)
    if oneprob > zeroprob:
        for i in range(index - delta):
            probs[i] = (i/(index - delta)) * zeroprob
        for i in range(index - delta, index):
            probs[i] = (((oneprob - zeroprob) / delta) * (i - index)) + oneprob
        for i in range(index, length):
            probs[i] = (((1 - oneprob) / (length-index)) * (i - index)) + oneprob
    else:
        for i in range(length):
            probs[i] = (i/length) * oneprob
    with open(os.path.join(outfolder, ID + ".psv"), 'w') as testfile:
        print('PredictedProbability|PredictedLabel', file=testfile)
        for P in probs:
            #print(str(P) + '|' + str(int(P>0.5)))
            print(str(P) + '|' + str(int(P>0.5)), file=testfile)
    #probs = 'SepsisLabel' + probs
    #np.savetxt(os.path.join(outfolder, ID + ".psv"), probs, delimiter=",")
def RunonTest(picklepath, model='PredReg'):
    data_phy = pickle.load(open(picklepath, 'rb'))
    for ID in data_phy['test_ids']:
        if model == 'PredReg':
            oneprob, index = Run(data_phy['data'][ID][0], data_phy['data'][ID][2])
            CreateOut(ID, oneprob, index, len(data_phy['data'][ID][0]), outfolder='Predictions')
        else:
            Runregular(data_phy['data'][ID][0],data_phy['data'][ID][2] , os.path.join('Predictions', ID + '.psv'))