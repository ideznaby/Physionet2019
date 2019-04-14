# -*- coding: utf-8 -*-

from evaluate_sepsis_score import evaluate_scores
from CreateOutput import RunonTest
from contextlib import closing
from zipfile import ZipFile, ZIP_DEFLATED
import os

def zipdir(basedir, archivename):
    assert os.path.isdir(basedir)
    with closing(ZipFile(archivename, "w", ZIP_DEFLATED)) as z:
        for root, dirs, files in os.walk(basedir):
            #NOTE: ignore empty directories
            for fn in files:
                absfn = os.path.join(root, fn)
                zfn = absfn[len(basedir)+len(os.sep):] #XXX: relative path
                z.write(absfn, zfn)
if __name__ == '__main__':
    picklepath = 'Physionet19_avg_glob_SetA.pkl'
    RunonTest(picklepath)
    zipdir('Predictions', 'Preds.zip')
    auroc, auprc, accuracy, f_measure, utility = evaluate_scores('Real.zip', 'Preds.zip')
    print(auroc, auprc, accuracy, f_measure, utility)
