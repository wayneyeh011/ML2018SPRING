import numpy as np
import pandas as pd
import csv
import sys


### This is a memo of Training ###
#   1. preprocessing
#   2. algo
#   3. Training
#   4. Prediction



## 1. Preprocessing

## 1-1 Import and Basic Clean
data_inc = pd.read_csv('../input/train.csv')
data_inc.tail(10) 

#  Strip (whitespace) of string
for col in ['workclass','education','marital_status','occupation','relationship','race','sex','native_country','income']:
    data_inc[col] = data_inc[col].str.strip()

# To Numeric 
for col in ['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']:
    data_inc[col] = pd.to_numeric(data_inc[col], errors='coerce')
    
#  Strange obs
data_inc.loc[(data_inc.loc[:,'sex'] == 'Male') & (data_inc.loc[:,'relationship'] == 'Wife'), 'relationship'] = np.nan          
data_inc.loc[(data_inc.loc[:,'sex'] == 'Female') & (data_inc.loc[:,'relationship'] == 'Husband'), 'relationship'] = np.nan       
data_inc = data_inc.dropna()
data_inc.shape

# set index since some obs dropped
data_inc['index'] = np.arange(32558)
data_inc = data_inc.set_index('index')



## 1-2 Catagorial Dummy and Cluster them

# Workclass  (Consider to drop 'c6' class ... too small) 
# .. alternatively, to have 'full' dummy...simply set dummy on the original col
# .. ? is c6
map1 = {'Self-emp-inc':'c1','Federal-gov':'c2','Local-gov':'c3','Self-emp-not-inc':'c3','State-gov':'c3','Private':'c4','Never-worked':'c5','Without-pay':'c5','?':'c6'}
data_inc['wc2'] = data_inc['workclass'].map(map1)
temp = pd.get_dummies(data_inc['wc2'], prefix='wc')
data_inc = pd.concat( [data_inc, temp],axis=1 )


# edu
map1 = {'Doctorate':'c1','Prof-school':'c2','Masters':'c3','Bachelors':'c4','Assoc-voc':'c5','Assoc-acdm':'c5','Some-college':'c6',
        'HS-grad':'c7','12th':'c8','11th':'c8','10th':'c8','9th':'c8','7th-8th':'c8','5th-6th':'c8','1st-4th':'c8','Preschool':'c8'}
data_inc['edu2'] = data_inc['education'].map(map1)
temp = pd.get_dummies(data_inc['edu2'], prefix='edu')
data_inc = pd.concat( [data_inc, temp],axis=1 )


# marital_status
map1 = {'Married-civ-spouse':'c1','Married-AF-spouse':'c1','Divorced':'c2','Married-spouse-absent':'c2','Widowed':'c2',
        'Separated':'c3','Never-married':'c4'}
data_inc['mtstat2'] = data_inc['marital_status'].map(map1)
temp = pd.get_dummies(data_inc['mtstat2'], prefix='mtstat')
data_inc = pd.concat( [data_inc, temp],axis=1 )


# occupation
# ? is c11
map1 = {'Exec-managerial':'c1','Prof-specialty':'c2','Protective-serv':'c3','Tech-support':'c3','Sales':'c4',
        'Craft-repair':'c5','Transport-moving':'c6','Adm-clerical':'c7','Machine-op-inspct':'c7','Farming-fishing':'c7','Armed-Forces':'c7',
        'Handlers-cleaners':'c8','Other-service':'c9','Priv-house-serv':'c10','?':'c11'}
data_inc['job2'] = data_inc['occupation'].map(map1)
temp = pd.get_dummies(data_inc['job2'], prefix='job')
data_inc = pd.concat( [data_inc, temp],axis=1 )


# Relationship
map1 = {'Husband':'c1','Wife':'c2','Not-in-family':'c3','Unmarried':'c4','Other-relative':'c5','Own-child':'c6'}
data_inc['re2'] = data_inc['relationship'].map(map1)
temp = pd.get_dummies(data_inc['re2'], prefix='re')
data_inc = pd.concat( [data_inc, temp],axis=1 )


# Race
map1 = {'White':'c1','Asian-Pac-Islander':'c1','Black':'c2','Amer-Indian-Eskimo':'c3','Other':'c4'}
data_inc['race2'] = data_inc['race'].map(map1)
temp = pd.get_dummies(data_inc['race2'], prefix='race')
data_inc = pd.concat( [data_inc, temp],axis=1 )


# Sex
map1 = {'Male':'c1','Female':'c2'}
data_inc['sex2'] = data_inc['sex'].map(map1)
temp = pd.get_dummies(data_inc['sex2'], prefix='sex')
data_inc = pd.concat( [data_inc, temp],axis=1 )


# native_country_United-States
# (1) Full (2) by continent (3) by GDP
map1 = {'United-States':'c1','Canada':'c1','Outlying-US(Guam-USVI-etc)':'c1',
        'England':'c2','Germany':'c2','Greece':'c2','Italy':'c2','Poland':'c2','Portugal':'c2','Ireland':'c2','France':'c2','Hungary':'c2','Scotland':'c2','Yugoslavia':'c2','Holand-Netherlands':'c2',
        'Puerto-Rico':'c3','Cuba':'c3','Honduras':'c3','Jamaica':'c3','Mexico':'c3','Dominican-Republic':'c3','Ecuador':'c3','Haiti':'c3','Columbia':'c3','Guatemala':'c3','Nicaragua':'c3','El-Salvador':'c3','Trinadad&Tobago':'c3','Peru':'c3',          
        'Cambodia':'c4','India':'c4','Japan':'c4','China':'c4','Iran':'c4','Philippines':'c4','Vietnam':'c4','Laos':'c4','Taiwan':'c4','Thailand':'c4','Hong':'c4','South':'c4','?':'c4'}
data_inc['country2'] = data_inc['native_country'].map(map1)
temp = pd.get_dummies(data_inc['country2'], prefix='country')
data_inc = pd.concat( [data_inc, temp],axis=1 )

## Maybe OECD?
#  Australia, Austria, Belgium, Canada, Chile, Czech Republic, Denmark, Estonia, Finland, France, Germany, Greece, Hungary, Iceland, Ireland, Israel, Italy, Japan, Korea, Luxembourg, Mexico, the Netherlands, New Zealand, Norway, Poland, Portugal, Slovak Republic, Slovenia, Spain, Sweden, Switzerland, Turkey, the United Kingdom, and the United States. 
map1 = {'United-States':'oecd','Canada':'oecd','Outlying-US(Guam-USVI-etc)':'oecd',
        'England':'oecd','Germany':'oecd','Greece':'oecd','Italy':'oecd','Poland':'oecd','Portugal':'oecd','Ireland':'oecd','France':'oecd','Hungary':'oecd','Scotland':'oecd','Yugoslavia':'not','Holand-Netherlands':'oecd',
        'Puerto-Rico':'not','Cuba':'not','Honduras':'not','Jamaica':'not','Mexico':'oecd','Dominican-Republic':'not','Ecuador':'not','Haiti':'not','Columbia':'not','Guatemala':'not','Nicaragua':'not','El-Salvador':'not','Trinadad&Tobago':'not','Peru':'not',          
        'Cambodia':'not','India':'not','Japan':'oecd','China':'not','Iran':'not','Philippines':'not','Vietnam':'not','Laos':'not','Taiwan':'not','Thailand':'not','Hong':'not','South':'oecd','?':'not'}
data_inc['country_oecd'] = data_inc['native_country'].map(map1)
temp = pd.get_dummies(data_inc['country_oecd'], prefix='oecd')
data_inc = pd.concat( [data_inc, temp],axis=1 )


# income
map1 = {'<=50K':0,'>50K':1}
data_inc['income'] = data_inc['income'].map(map1)




## 1-3 Additional Feature Trans(poly term, Interaction)

## Capital (and related)
data_inc['has_cap'] = ((data_inc['capital_gain'] > 0) | (data_inc['capital_loss'] > 0))*1
data_inc['has_capGain'] = (data_inc['capital_gain'] > 0)*1
data_inc['has_capLoss'] = (data_inc['capital_loss'] > 0)*1
data_inc['capFlow'] = (data_inc['capital_gain'] + data_inc['capital_loss'])*1
data_inc['capFlow_p2'] = data_inc['capFlow']**2
data_inc['capFlow_p3'] = data_inc['capFlow']**3

data_inc['many_cap'] = (data_inc['capFlow'] > 7000)*1
data_inc['mid_cap'] = ((data_inc['capFlow'] <= 7000) | (data_inc['capFlow'] > 0))*1
data_inc['no_cap'] = (data_inc['capFlow'] == 0)*1
#data_inc['cf2'] = pd.cut(data_inc['capFlow'],bins=[0.01,1000,2000,3000,4000,5000,7000,100000], labels=['c' + str(i) for i in range(1,8)])
#temp = pd.get_dummies(data_inc['cf2'], prefix='cf')
#data_inc = pd.concat( [data_inc, temp],axis=1 )


## Age
data_inc['age_p2'] = data_inc['age']**2
data_inc['age_p3'] = data_inc['age']**3


## edu_num, edu_num x female (This is an 'approximately' continuous var) 
data_inc['education_num_p2'] = data_inc['education_num'] ** 2
data_inc['education_num_p3'] = data_inc['education_num'] ** 3
data_inc['female_c_edu_p2'] = data_inc['sex_c2'] * (data_inc['education_num'] ** 2)
data_inc['female_c_edu'] = data_inc['sex_c2'] * data_inc['education_num']


# Discritize
#data_inc['d_age'] = pd.cut(data_inc['age'], bins=[0,20,25,30,35,40,45,50,55,60,100], labels=['c' + str(i) for i in range(1,11)])
#temp = pd.get_dummies(data_inc['d_age'], prefix='d_age')
#data_inc = pd.concat( [data_inc, temp],axis=1 )


## hours_per_week (and related)
data_inc['hours_per_week_p2'] = data_inc['hours_per_week']**2
data_inc['hours_per_week_p3'] = data_inc['hours_per_week']**3
data_inc['hours_per_week_p2'] = data_inc['hours_per_week']**2

# Discritize
data_inc['hr2'] = pd.cut(data_inc['hours_per_week'], bins=[0,32.5,37.5,42.5,47.5,52.5,57.5,62.5,67.5,100], labels=['c' + str(i) for i in range(1,10)])
temp = pd.get_dummies(data_inc['hr2'], prefix='hr')
data_inc = pd.concat( [data_inc, temp],axis=1 )


# fnlwgt ??? (It seems not important)
data_inc['fnlwgt_p2'] = data_inc['fnlwgt'] **2
data_inc['fnlwgt_p3'] = data_inc['fnlwgt'] **3


# Sex x Race
for col in ['race_c1', 'race_c2', 'race_c3', 'race_c4']:
    data_inc['Female_' + col] = data_inc['sex_c2'] * data_inc[col]

# Sex x Edu
for col in ['edu_c1', 'edu_c2', 'edu_c3', 'edu_c4', 'edu_c5', 'edu_c6','edu_c7', 'edu_c8']:
    data_inc['Female_' + col] = data_inc['sex_c2'] * data_inc[col]

# Sex x Occupation (with imputed version?)
for col in ['job_c1', 'job_c2', 'job_c3', 'job_c4', 'job_c5', 'job_c6','job_c7', 'job_c8', 'job_c9','job_c10','job_c11']:
    data_inc['Female_' + col] = data_inc['sex_c2'] * data_inc[col]

# Sex x Workclass
for col in ['wc_c1', 'wc_c2', 'wc_c3', 'wc_c4', 'wc_c5', 'wc_c6']:
    data_inc['Female_' + col] = data_inc['sex_c2'] * data_inc[col]    

# high edu x self-emp-not-inc 
data_inc['HighEdu'] = ((data_inc['edu2'] == 'c1') | (data_inc['edu2'] == 'c2') | (data_inc['edu2'] == 'c3'))*1
data_inc['HighEdu_SNINC'] = ((data_inc['HighEdu'] == 1) & (data_inc['workclass'] == 'Self-emp-not-inc'))*1



## 1-4 Saving

data_inc.to_csv('../output/mid.csv')
data_inc.shape








## 2. Imlemented algo

## (1) LOGIT_GD
def GD_Logit(X, y, w0, reg=0, eta=0.01, ada=True,conv_norm=0.1,max_force=100):  # Add regularizer
    
    '''
    For Logistic
    Note  -Initial GD could be large, this may influence GD.
          -Still, we need a good eta ... or GD may work badly
          -Adding regularizor, L1, L2? 
          -Use vectorized/itemize computation to speed up
    Additional...(1) logLike
    Loss func = -logLike + (regularization) 
    '''
    
    ## Prepare
    X, y, w0, dim = np.array(X), np.array(y), np.array(w0), len(w0)
    force = 0
    
    z_vec = X.dot(w0)
    sigmoid_vec = 1/(1 + np.exp(- z_vec))
    gd = X.T.dot(-(y-sigmoid_vec)) + 2*reg*w0     # -(y-sigmoid_vec)
    gd_sse, conv_log = gd**2, []                  # gd_log=[gd1,gd2,...]; conv_log=[norm1,norm2,...]
    conv_log.append(np.linalg.norm(gd))
    
    print (gd, gd_sse)
    ## Main...(print ('w0',w0), print ('gd',gd), )
    while np.linalg.norm(gd) >= conv_norm and force <= max_force:
        #if force % 500 == 0:  # Freq to report
        #    print ('GD current iter',force)
        #    #print ('Current gd and w', gd, w0)
        #    print ('Current gd', gd)
        force += 1        
        ada_size = np.array([1/np.sqrt(i) for i in gd_sse]) # Store the "Inverse Root SSE for each dim", updated in each iter
        #ada_size = 1/np.sqrt(gd_sse)

        if ada==True:
            w0 = w0 - eta*ada_size*gd
        else:
            w0 = w0 - eta*gd

        z_vec = X.dot(w0)
        sigmoid_vec = 1/(1 + np.exp(- z_vec))
        gd = X.T.dot(-(y-sigmoid_vec)) + 2*reg*w0
        gd_sse = gd_sse + gd**2               # gd_sse update (for ada)
        conv_log.append(np.linalg.norm(gd))   # log of convergence of gd
        
    print ('Final gd and w', gd, w0)
    return w0




## (2) bestThresh
def bestThresh(pred,true, low=0.4, up=0.6, step=0.001):
    
    pred, true = np.array(pred), np.array(true)    
    thresh = np.arange(low,up,step)   
    accuracy = []
    for i in thresh:
        pred2 = (pred > i)*1
        accuracy.append( ((pred2 == true)*1).sum() )
        #print (((pred2 == true)*1).sum())
    accuracy = np.array(accuracy)
    bestThresh = thresh[accuracy.argmax()]
    
    return [bestThresh, (pred > bestThresh)*1]





## (3) CV
def CVdraw(size, fold):
    permute = np.random.permutation(size)
    foldsize = size // fold 
    train, val = [], []
    print ('The approximated foldsize', foldsize)
    
    for i in range(fold):
        if i == fold - 1:
            val_index = permute[foldsize*i:]
            train_index = list(set(range(size)) - set(val_index))
            val.append(val_index)
            train.append(train_index)            
        else:
            val_index = permute[foldsize*(i):foldsize*(i+1)]
            train_index = list(set(range(size)) - set(val_index))
            val.append(val_index)
            train.append(train_index)
    
    return {'fold': fold, 'train':train, 'val':val}




## (4) Training

def Train(X,y,cv,eta1=0.01,eta2=0.01,reg=0,supCV=False,supFin=False):  # supCV=False,supFin=False
    
    '''
    My "Train" is designed as follow: 
    ''' 
    
    X, y = np.array(X), np.array(y)
    folds, datasize, dim = cv['fold'], X.shape[0], X.shape[1]
    report = np.repeat(0,folds) 
    
    if supCV == False:
        for k in range(folds):     
            ## Train
            tr_X = X[cv['train'][k],:]
            tr_y = y[[cv['train'][k]]]
            w0 = np.repeat(0,dim)
            w = GD_Logit(tr_X, tr_y, w0, max_force=1000,reg=reg,eta=eta1)  # eta, ...
            #w = logit(tr_y, tr_X, eta1, 1000)  # Tien's Version
            tr_prob = 1/(1+np.exp(-tr_X.dot(w)))
            tau = bestThresh(tr_prob, tr_y)[0]            # Use (sub)training set to choose best threshold(tau)

            ## Evaluation --- by accuracy
            val_X = X[cv['val'][k],:]
            val_y = y[[cv['val'][k]]]
            val_prob = 1/(1+np.exp(-val_X.dot(w)))
            val_pred = (val_prob >= tau)*1    # Array
            correct_num = ((val_pred == val_y)*1).sum()
            report[k] = correct_num
            
        # Final Model...exploit whole data 
        print ('CV done')
        accuracy = (report.sum())/datasize
        
    if supFin == False:
        all_X, all_y = X, y
        w0 = np.repeat(0,dim)
        w = GD_Logit(X,y,w0, max_force=10000, reg=reg,eta=eta2)  # Modify final model training
        #w = logit(tr_y, tr_X, eta2, 10000)
        all_prob = 1/(1+np.exp(-all_X.dot(w)))
        tau = bestThresh(all_prob,all_y,step=0.0001)[0]
        all_pred = (all_prob >= tau)*1 
        accuracy2 = ((all_pred == all_y)*1).sum()/datasize
        
    if supCV == False and supFin == False:
        return {'finModel':w,'finTau':tau,'finProb':all_prob,'finPred':all_pred,'val_acc':accuracy,'all_acc':accuracy2}
    elif supCV == True:
        return accuracy2
    elif supFin == True:
        return accuracy




## (5) Generative model

def LinDisc(X,y,sameCov=True):
    
    '''
    1. y should be in {0,1}
    2. Demean matrix: I - 1/n(11'); where 1 is a vector with all entries = 1
       --- simply use np.cov (to adjust bias or not...)
    3. Q...should Sig1, Sig2 (1) avg b/t Sig1 Sig2 (2) Compute at the same time
    '''
    ## Prepare
    X, y= np.array(X), np.array(y)
    dim, size = X.shape[1], X.shape[0] 
    
    ## Size, Mean, SD
    n0 = len(np.where(y==0)[0])
    n1 = len(np.where(y==1)[0])    
    u0 = X[np.where(y==0)[0],].mean(axis=0)
    u1 = X[np.where(y==1)[0],].mean(axis=0)
    
    if sameCov==True:
        Sig = np.cov(X.T)
        #print (Sig.shape)
        iSig = np.linalg.inv(Sig) 
        zval = lambda xvec: -0.5*((xvec-u1).dot(iSig).dot(xvec-u1) - (xvec-u0).dot(iSig).dot(xvec-u0)) + np.log(n1/n0)      
        prob1 = np.array( [1/(1 + np.exp(-zval(X[i,]))) for i in range(size)] )        
        pred = (prob1 >= 0.5)*1
        return {'pred':pred, 'Sig_hat':Sig, 'mu0_hat':u0, 'mu1_hat': u1}
        
    else:
        Sig0 = np.cov(X[np.where(y==0)[0],].T)
        Sig1 = np.cov(X[np.where(y==1)[0],].T)
        det0, det1 = np.linalg.det(Sig0), np.linalg.det(Sig1) 
        iSig0, iSig1 = np.linalg.inv(Sig0), np.linalg.inv(Sig1)
        
        zval = lambda xvec: np.log(np.sqrt(det0/det1)) -0.5*((xvec-u1).dot(iSig1).dot(xvec-u1) - (xvec-u0).dot(iSig0).dot(xvec-u0)) + np.log(n1/n0)      
        prob1 = np.array( [1/(1 + np.exp(-zval(X[i,]))) for i in range(size)] )
        pred = (prob1 >= 0.5)*1
        return {'pred':pred, 'Sig0_hat':Sig0,'Sig1_hat':Sig1,'mu0_hat':u0, 'mu1_hat': u1}








## 3. Training


## 3-1 Draw CV
cv1 = CVdraw(32558, 5)

## 3-2 Train
data_inc['cept'] = 1

# Mod 3 (discrete hr, many cap)*
feature_list = (['cept','age','age_p2','age_p3'] + ['wc_c' + str(i) for i in range(2,7)] + ['edu_c' + str(i) for i in range(2,9)] +  
    ['mtstat_c' + str(i) for i in range(2,5)] + ['job_c' + str(i) for i in range(2,12)] + ['re_c' + str(i) for i in range(2,7)] + 
    ['race_c' + str(i) for i in range(2,5)] + ['sex_c2'] + ['country_c' + str(i) for i in range(2,5)] + ['has_capGain','has_capLoss','capFlow','capFlow_p2','capFlow_p3','many_cap'] +
    ['hr_c' + str(i) for i in range(2,10)] + ['fnlwgt'] + ['Female_race_c' + str(i) for i in range(2,5)] + ['Female_edu_c' + str(i) for i in range(2,9)] +
    ['oecd_oecd'] + ['Female_job_c' + str(i) for i in range(2,12)])


## 3-3 Normalization
for col in ['age','age_p2','age_p3','capFlow','capFlow_p2','capFlow_p3','hours_per_week','hours_per_week_p2','hours_per_week_p3','fnlwgt','fnlwgt_p2','fnlwgt_p3']:
    data_inc[col] = (data_inc[col] - data_inc[col].mean())/data_inc[col].std()


## 3-4 Train
Xmat = np.array(data_inc[feature_list])
yvec = np.array(data_inc['income'])
w0 = np.repeat(0,len(feature_list))

result = Train(Xmat,yvec,cv1,eta1=0.2,eta2=0.2, reg=0.5) # supFin ,supCV=True, reg=0.1







## 4. Predicting 

## 4-1 Read Data

sysarg = '../input/test.csv'
text = open(sysarg,"r")
row = csv.reader(text , delimiter= ",")

testset = []
for r in row:
    testset.append(r)
text.close()
col_list = testset[0]
testset = testset[1:]
testset = np.array(testset)
#print (testset.shape)

testset = pd.DataFrame(testset)
testset.columns = col_list
testset.head()




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
# (1) Full (2) by continent (3) by GDP
map1 = {'United-States':'c1','Canada':'c1','Outlying-US(Guam-USVI-etc)':'c1',
        'England':'c2','Germany':'c2','Greece':'c2','Italy':'c2','Poland':'c2','Portugal':'c2','Ireland':'c2','France':'c2','Hungary':'c2','Scotland':'c2','Yugoslavia':'c2','Holand-Netherlands':'c2',
        'Puerto-Rico':'c3','Cuba':'c3','Honduras':'c3','Jamaica':'c3','Mexico':'c3','Dominican-Republic':'c3','Ecuador':'c3','Haiti':'c3','Columbia':'c3','Guatemala':'c3','Nicaragua':'c3','El-Salvador':'c3','Trinadad&Tobago':'c3','Peru':'c3',          
        'Cambodia':'c4','India':'c4','Japan':'c4','China':'c4','Iran':'c4','Philippines':'c4','Vietnam':'c4','Laos':'c4','Taiwan':'c4','Thailand':'c4','Hong':'c4','South':'c4','?':'c4'}
testset['country2'] = testset['native_country'].map(map1)
temp = pd.get_dummies(testset['country2'], prefix='country')
testset = pd.concat( [testset, temp],axis=1 )

# OECD country
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
testset['mid_cap'] = ((testset['capFlow'] <= 7000) | (testset['capFlow'] > 0))*1
testset['no_cap'] = (testset['capFlow'] == 0)*1


# Age
testset['age_p2'] = testset['age']**2
testset['age_p3'] = testset['age']**3


## edu_num, edu_num x female (This is an 'approximately' continuous var) 
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
    
# White x Occupation
for col in ['job_c1', 'job_c2', 'job_c3', 'job_c4', 'job_c5', 'job_c6','job_c7', 'job_c8', 'job_c9','job_c10','job_c11']:
    testset['White_' + col] = testset['race_c1'] * testset[col]
    
# highedu x self-not-emp-inc
testset['HighEdu'] = ((testset['edu2'] == 'c1') | (testset['edu2'] == 'c2') | (testset['edu2'] == 'c3'))*1
testset['HighEdu_SNINC'] = ((testset['HighEdu'] == 1) & (testset['workclass'] == 'Self-emp-not-inc'))*1




## 4-3 Normalization (by test)

for col in ['age','age_p2','age_p3','capFlow','capFlow_p2','capFlow_p3','hours_per_week','hours_per_week_p2','hours_per_week_p3','fnlwgt','fnlwgt_p2','fnlwgt_p3']:
    testset[col] = (testset[col] - testset[col].mean())/testset[col].std()





## 4-4 Prediction   !!! ------ feature_list, tau required ------- !!!

# predicting
testset['cept'] = 1
testset = testset[feature_list]
print (len(testset.columns), len(feature_list),testset.columns)


coef = np.array(result['finModel'])
Threshold = result['finTau'] # 0.4577  # 0.5 0.4577  #result['finTau']
X = np.array(testset)
z_vec = X.dot(coef)
prob = 1/(1 + np.exp(-z_vec))   # Prob from logit, 1/1+exp(-w'x)
pred = (prob >= Threshold)*1    # The Predicted results
print (prob[:5],pred[:5])


# exporting
filename = '../output/compare_attr'   # sys.argv[]
f1 = open(filename, 'w+')
f1_text = csv.writer(f1,delimiter=',',lineterminator='\n')
f1_text.writerow(['id','label'])
for i in range(testset.shape[0]):
    f1_text.writerow([int(i) + 1, pred[i]]) # pred[i]
f1.close()



















