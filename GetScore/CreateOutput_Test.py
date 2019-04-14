import pickle
import os
from CreateOutput import CreateOut

data_phy = pickle.load(open('Preprocessing/Physionet19_avg_glob_SetA.pkl', 'rb'))
for ID in data_phy['test_ids']:
    binlabel = data_phy['data'][ID][4]
    index = data_phy['data'][ID][5]
    length = len(data_phy['data'][ID][0])
    labels = np.array(data_phy['data'][ID][3], np.int32)
   # labels = np.insert(labels, 0, 'SepsisLabel')
    #labels = 'SepsisLabel' + labels
    with open(os.path.join('Dummy/TestPreds', ID + ".psv"), 'w') as testfile:
        print('SepsisLabel', file=testfile)
        for L in labels:
            print(L, file=testfile)
    if binlabel == 1:
        binarray = [0.1,0.9]
    else:
        binarray = [0.9,0.1]
    CreateOut(ID, binarray, index, length, delta=5, outfolder = 'Dummy/DummyPreds')