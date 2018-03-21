## 0.Load Final Model
import sys
import csv 
import pandas as pd
import numpy as np



## 1. Read
filename = sys.argv[1]
text = open(filename,"r")
row = csv.reader(text , delimiter= ",")

testset = []
for r in row:
    testset.append(r)
text.close()
testset = np.array(testset)
testset.reshape([4680,11])
testset = pd.DataFrame(testset)
testset.columns = ['id','測項','hr_9','hr_8','hr_7','hr_6','hr_5','hr_4','hr_3','hr_2','hr_1']

testset = testset.loc[:,['id','測項'] + ['hr_' + str(i) for i in range(1,10)]]
testset = testset.set_index(['id','測項']).stack().unstack(['測項'])




## 2. Pre-Processing
#  Clean
testset.loc[testset.loc[:,'RAINFALL'] == 'NR', 'RAINFALL'] = '0'
for var in testset.columns:
    testset.loc[:,var] = pd.to_numeric(testset.loc[:,var],errors='coerce')
    testset.loc[testset.loc[:,var] < 0, var] = 0
    
    
#  Wind
temp = (testset['WIND_DIREC']*(-1)+90)*np.pi/180  # map [0,360] to corresponding radian (ex. 90degree -> 1.57 (pi/2))
xtemp = testset['WIND_SPEED']*np.cos(temp)        # WindSP to x vector
ytemp = testset['WIND_SPEED']*np.sin(temp)        # WindSP to y vector
testset['East'] = (xtemp >= 0)*1
testset['South'] = (ytemp <= 0)*1
testset['xvec'] = np.abs(xtemp)
testset['yvec'] = np.abs(ytemp)
testset['xvec_E'] = np.abs(xtemp) * np.array(xtemp >= 0)
testset['yvec_S'] = np.abs(ytemp) * np.array(ytemp <= 0)

temp = (testset['WD_HR']*(-1)+90)*np.pi/180
xtemp = testset['WS_HR']*np.cos(temp)
ytemp = testset['WS_HR']*np.sin(temp)
testset['Easthr'] = (xtemp >= 0)*1
testset['Southhr'] = (ytemp <= 0)*1
testset['xvechr'] = np.abs(xtemp)
testset['yvechr'] = np.abs(ytemp)
testset['xvec_Ehr'] = np.abs(xtemp) * np.array(xtemp >= 0)
testset['yvec_Shr'] = np.abs(ytemp) * np.array(ytemp <= 0)


# Imputing strange value
temphr = []
for i in range(260):
    temphr += [1,2,3,4,5,6,7,8,9]
testset['temphr'] = temphr

for col in ['AMB_TEMP','CH4','CO','NMHC','NO','NO2','NOx','O3','PM10','PM2.5','RH','SO2','THC','xvec','yvec','xvechr','yvechr']:
    testset.loc[(testset.loc[:,col] == 0) & (testset.loc[:,'temphr'] != 9), col] = np.nan
    testset.loc[:,col] = testset.loc[:,col].fillna(method='bfill')
    testset.loc[testset.loc[:,col] == 0, col] = np.nan
    testset.loc[:,col] = testset.loc[:,col].fillna(method='ffill')
del testset['temphr']





## 3. Lag Term and Additional Features

# Lag terms
lag_list = ['AMB_TEMP','CH4','CO','NMHC','NO','NO2','NOx','O3','PM10','PM2.5','RAINFALL','RH','SO2','THC','East','South','xvec','yvec','xvec_E','yvec_S','Southhr','Easthr','xvechr','yvechr','xvec_Ehr', 'yvec_Shr']  
testset = testset[lag_list]
testset.index.names = ['id','hr']
testset = testset.unstack('hr')
new_col = []
for item in lag_list:
    new_col += [item + '_' + str(i) for i in range(1,10)]

testset.columns = new_col

# To sort by id
testset = testset.reset_index()
idindex = testset['id'].str.split('_')
idindex = [int(i[1]) for i in idindex]
testset['idindex'] = idindex
testset = testset.sort_values(by=['idindex'])
testset = testset.set_index('id')


# 2th term, 9hr mean
for col in lag_list:
    testset[col + '_1_p2'] = testset[col + '_1'] ** 2
    testset[col + '_mean'] = (testset[col + '_1'] + testset[col + '_2'] + testset[col + '_3'] + testset[col + '_4'] + testset[col + '_5'] + testset[col + '_6'] + testset[col + '_7'] + testset[col + '_8'] + testset[col + '_9']) / 9

# No Rain 9hr
testset['noRain9hr'] = (testset['RAINFALL_mean'] == 0)*1

# wind, rain interaction
testset['windrainx'] = testset['RAINFALL_1'] * testset['xvec_1']
testset['windrainy'] = testset['RAINFALL_1'] * testset['yvec_1']




## 4. Prediction 
testset['cept'] = 1
feature_list = (['cept','CH4_1','CH4_1_p2','CO_1','CO_1_p2','NO_mean','NO2_mean','NOx_1','NOx_mean','O3_1','O3_2','PM10_1','PM2.5_1','noRain9hr','RAINFALL_mean','RAINFALL_1','RAINFALL_2','RAINFALL_3','RAINFALL_4','windrainx','windrainy'] + 
     ['RH_1','SO2_1','SO2_1_p2','THC_1','THC_1_p2','Easthr_1','Southhr_1','Southhr_mean','xvechr_1','xvechr_mean','yvechr_1','xvec_Ehr_1','yvec_Shr_1','East_1','South_1','xvec_1','yvec_1','xvec_E_1','yvec_S_1'])
testset = testset[feature_list]


# predicting
testset = testset[feature_list]
coef = np.load('model_m3fin.npy')

pred = []
for i in range(testset.shape[0]):
    xvec = np.array(testset.loc['id_' + str(i)])
    ans = xvec.dot(coef)
    pred.append(ans)


# exporting
filename = sys.argv[2]
f1 = open(filename, 'w+')
f1_text = csv.writer(f1,delimiter=',',lineterminator='\n')

f1_text.writerow(['id','value'])
for i in range(testset.shape[0]):
    f1_text.writerow(['id_'+str(i), pred[i]]) 
f1.close()
