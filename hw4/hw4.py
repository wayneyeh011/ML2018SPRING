import numpy as np
import pandas as pd
import csv
import sys




## 0. Read Me
#  (1) This is a simplified hw4 (answering) file
#  (2) A full process please refer to hw4_train.py




## 1. Predicting  --- map from 
test_case = pd.read_csv(sys.argv[2])  # sys.argv[2]  '../input/test_case.csv'  
test_case = test_case.sort_values(['ID'])
test_case = np.array(test_case)

#
clabel = np.load('myfin_clabel.npy')
#

filename = sys.argv[3]   # sys.argv[]  '../output/cae_pcnor_km'
f1 = open(filename, 'w+')
f1_text = csv.writer(f1,delimiter=',',lineterminator='\n')
f1_text.writerow(['ID','Ans'])
for i in range(test_case.shape[0]):
    id1, id2 = test_case[i,1], test_case[i,2]
    ans = (clabel[id1] == clabel[id2])*1
    f1_text.writerow([int(i), ans]) # pred[i]
f1.close()

