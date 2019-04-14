# -*- coding: utf-8 -*-
#This file is a template for prediction and regression models which return a clasisification label and a regression value for the index
#This method takes the data and returns two values: the probability of sepsis happening and the index of that happening
def Run(indata, covs):
    prob = 0.8
    index = len(indata) - 2
    return prob, index