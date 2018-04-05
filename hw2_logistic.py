import numpy as np
import pandas as pd
import csv
import sys


## NOTE ... model 'mod3_iter1w.npy'


## 4-1 Read Data  

sysarg = sys.argv[2]  #'../input/test.csv'
text = open(sysarg,"r")
row = csv.reader(text , delimiter= ",")

testset = []
for r in row:
    testset.append(r)
text.close()
col_list = testset[0]
testset = testset[1:]
testset = np.array(testset)
testset = pd.DataFrame(testset)
testset.columns = col_list




## 4-2 Clean, gen features, ...
## Strip string
for col in ['workclass','education','marital_status','occupation','relationship','race','sex','native_country']:
    testset[col] = testset[col].str.strip()
## To Numeric 
for col in ['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']:
    testset[col] = pd.to_numeric(testset[col], errors='coerce')


## gen dummy
# Workclass  (Consider to drop 'c6' class ... too small) 
map1 = {'Self-emp-inc':'c1','Federal-gov':'c2','Local-gov':'c3','Self-emp-not-inc':'c3','State-gov':'c3','Private':'c4','?':'c5','Never-worked':'c6','Without-pay':'c6'}
testset['wc2'] = testset['workclass'].map(map1)
temp = pd.get_dummies(testset['wc2'], prefix='wc')
testset = pd.concat( [testset, temp],axis=1 )

# edu
map1 = {'Doctorate':'c1','Prof-school':'c2','Masters':'c3','Bachelors':'c4','Assoc-voc':'c5','Assoc-acdm':'c5','Some-college':'c6',
        'HS-grad':'c7','12th':'c8','11th':'c8','10th':'c8','9th':'c8','7th-8th':'c8','5th-6th':'c8','1st-4th':'c8','Preschool':'c8'}
testset['edu2'] = testset['education'].map(map1)
temp = pd.get_dummies(testset['edu2'], prefix='edu')
testset = pd.concat( [testset, temp],axis=1 )

# marital_status
map1 = {'Married-civ-spouse':'c1','Married-AF-spouse':'c1','Divorced':'c2','Married-spouse-absent':'c2','Widowed':'c2',
        'Separated':'c3','Never-married':'c4'}
testset['mtstat2'] = testset['marital_status'].map(map1)
temp = pd.get_dummies(testset['mtstat2'], prefix='mtstat')
testset = pd.concat( [testset, temp],axis=1 )

# occupation
map1 = {'Exec-managerial':'c1','Prof-specialty':'c2','Protective-serv':'c3','Tech-support':'c3','Sales':'c4',
        'Craft-repair':'c5','Transport-moving':'c6','Adm-clerical':'c7','Machine-op-inspct':'c7','Farming-fishing':'c7','Armed-Forces':'c7',
        '?':'c8','Handlers-cleaners':'c9','Other-service':'c10','Priv-house-serv':'c11'}
testset['job2'] = testset['occupation'].map(map1)
temp = pd.get_dummies(testset['job2'], prefix='job')
testset = pd.concat( [testset, temp],axis=1 )

# Relationship
map1 = {'Husband':'c1','Wife':'c2','Not-in-family':'c3','Unmarried':'c4','Other-relative':'c5','Own-child':'c6'}
testset['re2'] = testset['relationship'].map(map1)
temp = pd.get_dummies(testset['re2'], prefix='re')
testset = pd.concat( [testset, temp],axis=1 )

# Race
map1 = {'White':'c1','Asian-Pac-Islander':'c1','Black':'c2','Amer-Indian-Eskimo':'c3','Other':'c4'}
testset['race2'] = testset['race'].map(map1)
temp = pd.get_dummies(testset['race2'], prefix='race')
testset = pd.concat( [testset, temp],axis=1 )

# Sex
map1 = {'Male':'c1','Female':'c2'}
testset['sex2'] = testset['sex'].map(map1)
temp = pd.get_dummies(testset['sex2'], prefix='sex')
testset = pd.concat( [testset, temp],axis=1 )

# native_country_United-States
# (2) by continent (3) by GDP
map1 = {'United-States':'c1','Canada':'c1','Outlying-US(Guam-USVI-etc)':'c1',
        'England':'c2','Germany':'c2','Greece':'c2','Italy':'c2','Poland':'c2','Portugal':'c2','Ireland':'c2','France':'c2','Hungary':'c2','Scotland':'c2','Yugoslavia':'c2','Holand-Netherlands':'c2',
        'Puerto-Rico':'c3','Cuba':'c3','Honduras':'c3','Jamaica':'c3','Mexico':'c3','Dominican-Republic':'c3','Ecuador':'c3','Haiti':'c3','Columbia':'c3','Guatemala':'c3','Nicaragua':'c3','El-Salvador':'c3','Trinadad&Tobago':'c3','Peru':'c3',          
        'Cambodia':'c4','India':'c4','Japan':'c4','China':'c4','Iran':'c4','Philippines':'c4','Vietnam':'c4','Laos':'c4','Taiwan':'c4','Thailand':'c4','Hong':'c4','South':'c4','?':'c4'}
testset['country2'] = testset['native_country'].map(map1)
temp = pd.get_dummies(testset['country2'], prefix='country')
testset = pd.concat( [testset, temp],axis=1 )

map1 = {'United-States':'oecd','Canada':'oecd','Outlying-US(Guam-USVI-etc)':'oecd',
        'England':'oecd','Germany':'oecd','Greece':'oecd','Italy':'oecd','Poland':'oecd','Portugal':'oecd','Ireland':'oecd','France':'oecd','Hungary':'oecd','Scotland':'oecd','Yugoslavia':'not','Holand-Netherlands':'oecd',
        'Puerto-Rico':'not','Cuba':'not','Honduras':'not','Jamaica':'not','Mexico':'oecd','Dominican-Republic':'not','Ecuador':'not','Haiti':'not','Columbia':'not','Guatemala':'not','Nicaragua':'not','El-Salvador':'not','Trinadad&Tobago':'not','Peru':'not',          
        'Cambodia':'not','India':'not','Japan':'oecd','China':'not','Iran':'not','Philippines':'not','Vietnam':'not','Laos':'not','Taiwan':'not','Thailand':'not','Hong':'not','South':'oecd','?':'not'}
testset['country_oecd'] = testset['native_country'].map(map1)
temp = pd.get_dummies(testset['country_oecd'], prefix='oecd')
testset = pd.concat( [testset, temp],axis=1 )



## Additional Features
# Capital
testset['has_cap'] = ((testset['capital_gain'] > 0) | (testset['capital_loss'] > 0))*1
testset['has_capGain'] = (testset['capital_gain'] > 0)*1
testset['has_capLoss'] = (testset['capital_loss'] > 0)*1
testset['capFlow'] = (testset['capital_gain'] + testset['capital_loss'])*1
testset['capFlow_p2'] = testset['capFlow']**2
testset['capFlow_p3'] = testset['capFlow']**3
testset['many_cap'] = (testset['capFlow'] > 7000)*1


# Age
testset['age_p2'] = testset['age']**2
testset['age_p3'] = testset['age']**3

# edu_num, edu_num x female (This is an 'approximately' continuous var) 
testset['education_num_p2'] = testset['education_num'] ** 2
testset['education_num_p3'] = testset['education_num'] ** 3
testset['female_c_edu_p2'] = testset['sex_c2'] * (testset['education_num'] ** 2)
testset['female_c_edu'] = testset['sex_c2'] * testset['education_num']

# fnlwgt
testset['fnlwgt_p2'] = testset['fnlwgt'] **2
testset['fnlwgt_p3'] = testset['fnlwgt'] **3

# hours_per_week, and discritized hr
testset['hours_per_week_p2'] = testset['hours_per_week']**2
testset['hours_per_week_p3'] = testset['hours_per_week']**3
testset['hr2'] = pd.cut(testset['hours_per_week'], bins=[0,32.5,37.5,42.5,47.5,52.5,57.5,62.5,67.5,100], labels=['c' + str(i) for i in range(1,10)])
temp = pd.get_dummies(testset['hr2'], prefix='hr')
testset = pd.concat( [testset, temp],axis=1 )


## Interaction
# Sex x Race
for col in ['race_c1', 'race_c2', 'race_c3', 'race_c4']:
    testset['Female_' + col] = testset['sex_c2'] * testset[col]
    
# Sex x Edu
for col in ['edu_c1', 'edu_c2', 'edu_c3', 'edu_c4', 'edu_c5', 'edu_c6','edu_c7', 'edu_c8']:
    testset['Female_' + col] = testset['sex_c2'] * testset[col]

# Sex x Occupation
for col in ['job_c1', 'job_c2', 'job_c3', 'job_c4', 'job_c5', 'job_c6','job_c7', 'job_c8', 'job_c9','job_c10','job_c11']:
    testset['Female_' + col] = testset['sex_c2'] * testset[col]

# Sex x workclass
for col in ['wc_c1', 'wc_c2', 'wc_c3', 'wc_c4', 'wc_c5', 'wc_c6']:
    testset['Female_' + col] = testset['sex_c2'] * testset[col]    

# highedu x self-not-emp-inc
testset['HighEdu'] = ((testset['edu2'] == 'c1') | (testset['edu2'] == 'c2') | (testset['edu2'] == 'c3'))*1
testset['HighEdu_SNINC'] = ((testset['HighEdu'] == 1) & (testset['workclass'] == 'Self-emp-not-inc'))*1





## 4-3 Normalization (by test)
for col in ['age','age_p2','age_p3','capFlow','capFlow_p2','capFlow_p3','hours_per_week','hours_per_week_p2','hours_per_week_p3','fnlwgt','fnlwgt_p2','fnlwgt_p3']:
    testset[col] = (testset[col] - testset[col].mean())/testset[col].std()






## 4-4 Prediction
# predicting
testset['cept'] = 1
feature_list = (['cept','age','age_p2','age_p3'] + ['wc_c' + str(i) for i in range(2,7)] + ['edu_c' + str(i) for i in range(2,9)] +  
    ['mtstat_c' + str(i) for i in range(2,5)] + ['job_c' + str(i) for i in range(2,12)] + ['re_c' + str(i) for i in range(2,7)] + 
    ['race_c' + str(i) for i in range(2,5)] + ['sex_c2'] + ['country_c' + str(i) for i in range(2,5)] + ['has_capGain','has_capLoss','capFlow','capFlow_p2','capFlow_p3','many_cap'] +
    ['hr_c' + str(i) for i in range(2,10)] + ['fnlwgt'] + ['Female_race_c' + str(i) for i in range(2,5)] + ['Female_edu_c' + str(i) for i in range(2,9)] +
    ['oecd_oecd'] + ['Female_job_c' + str(i) for i in range(2,12)])
testset = testset[feature_list]


coef = np.load('mod3_iter1w_reg05.npy')
Threshold = 0.4577 
X = np.array(testset)
z_vec = X.dot(coef)
prob = 1/(1 + np.exp(-z_vec))   # Prob from logit, 1/1+exp(-w'x)
pred = (prob >= Threshold)*1    # The Predicted results


# exporting
filename = sys.argv[6] #'../output/practice' # sys.argv[]
f1 = open(filename, 'w+')
f1_text = csv.writer(f1,delimiter=',',lineterminator='\n')
f1_text.writerow(['id','label'])
for i in range(testset.shape[0]):
    f1_text.writerow([int(i) + 1, pred[i]]) # pred[i]
f1.close()













