

## PART 0 READ ME ##
#  This is the "almost" complete record of training part 
#  I use specific setting when conducting training with may implementing func
#  If Somebody read this file, please tolerate my messy code. (I do not have solid engineering training) 
#  Organization as follow...PART 1 is preprocessing, PART 2 is some method I implmented and PART 3 is final training
#    - By R06323011 經濟所 葉政維





## PART 1 Preprocessing
##  1-1. Read, Reshape
pm25 = pd.read_csv('../input/train.csv', encoding='big5')
del pm25['測站']
pm25 = pm25.set_index(['日期','測項']).stack().unstack('測項')



## 1-2 Clean, Necessary transform, some time indicator
# Clean
pm25.loc[pm25.loc[:,'RAINFALL'] == 'NR', 'RAINFALL'] = '0'
for var in pm25.columns:
    pm25.loc[:,var] = pd.to_numeric(pm25.loc[:,var],errors='coerce')
    pm25.loc[pm25.loc[:,var] < 0, var] = 0

# Date, Season
pm25 = pm25.reset_index('日期')
dd = pd.DatetimeIndex(pm25.loc[:,'日期'])
pm25['Month'] = dd.month
pm25['Day'] = dd.day

pm25['Spring'] = (pm25['Month'] == 3) | (pm25['Month'] == 4) | (pm25['Month'] == 5)
pm25['Summer'] = (pm25['Month'] == 6) | (pm25['Month'] == 7) | (pm25['Month'] == 8)
pm25['Fall'] = (pm25['Month'] == 9) | (pm25['Month'] == 10) | (pm25['Month'] == 11)
pm25['Winter'] = (pm25['Month'] == 12) | (pm25['Month'] == 1) | (pm25['Month'] == 2)
pm25['Season'] = 1*pm25['Spring'] + 2*pm25['Summer'] + 3*pm25['Fall'] + 4*pm25['Winter']

# wind vector (and related)
temp = (pm25['WIND_DIREC']*(-1)+90)*np.pi/180  # map [0,360] to corresponding radian (ex. 90degree -> 1.57 (pi/2))
xtemp = pm25['WIND_SPEED']*np.cos(temp)        # WindSP to x vector
ytemp = pm25['WIND_SPEED']*np.sin(temp)        # WindSP to y vector
pm25['East'] = (xtemp >= 0)*1
pm25['South'] = (ytemp <= 0)*1
pm25['xvec'] = np.abs(xtemp)
pm25['yvec'] = np.abs(ytemp)
pm25['xvec_E'] = np.abs(xtemp) * np.array(xtemp >= 0)
pm25['yvec_S'] = np.abs(ytemp) * np.array(ytemp <= 0)

temp = (pm25['WD_HR']*(-1)+90)*np.pi/180
xtemp = pm25['WS_HR']*np.cos(temp)
ytemp = pm25['WS_HR']*np.sin(temp)
pm25['Easthr'] = (xtemp >= 0)*1
pm25['Southhr'] = (ytemp <= 0)*1
pm25['xvechr'] = np.abs(xtemp)
pm25['yvechr'] = np.abs(ytemp)
pm25['xvec_Ehr'] = np.abs(xtemp) * np.array(xtemp >= 0)
pm25['yvec_Shr'] = np.abs(ytemp) * np.array(ytemp <= 0)

## Imputing Suspious values
# RAINFALL...70, 1 obs
pm25.loc[pm25['RAINFALL'] > 50, 'RAINFALL'] = np.nan
pm25.loc[:,'RAINFALL'] = pm25.loc[:,'RAINFALL'].fillna(method='ffill')

# NMHC...4.83, 1 obs
pm25.loc[pm25['NMHC'] > 4, 'NMHC'] = np.nan
pm25.loc[:,'NMHC'] = pm25.loc[:,'NMHC'].fillna(method='ffill')

# temporature == 0...kind of strange
pm25.loc[pm25['AMB_TEMP'] == 0, 'AMB_TEMP'] = np.nan
pm25.loc[:,'AMB_TEMP'] = pm25.loc[:,'AMB_TEMP'].fillna(method='ffill')

# ... impute zeros ... it may be a (systematical) measure error
for col in ['AMB_TEMP','CH4','CO','NMHC','NO','NO2','NOx','O3','PM10','PM2.5','RH','SO2','THC','xvec','yvec','xvechr','yvechr']: #...     
    pm25.loc[pm25[col] == 0, col] = np.nan
    pm25.loc[:,col] = pm25.loc[:,col].fillna(method='ffill')




## 1-3 Produce lag term, and additional features
lag_list = ['AMB_TEMP','CH4','CO','NMHC','NO','NO2','NOx','O3','PM10','PM2.5','RAINFALL','RH','SO2','THC','East','South','xvec','yvec','xvec_E','yvec_S','Southhr','Easthr','xvechr','yvechr','xvec_Ehr', 'yvec_Shr']  # ,'WIND_SPEED','WS_HR'
pm25 = pm25.sort_values(by=['Month','Day'])
print (lag_list)

finalpm25 = []
for m in range(1,13):     
    mpm25 = pm25[pm25['Month'] == m]
    for col in lag_list:
        for lag in range(1,10):
            temp = pd.DataFrame(mpm25.loc[:,col].shift(lag))        
            temp.columns = [str(col) + '_' + str(lag)]
            mpm25 = pd.concat([mpm25, temp], axis=1)
    finalpm25.append(mpm25)

print (len(finalpm25), finalpm25[0].columns)
pm25 = pd.concat(finalpm25).dropna()




## 1-4 Additional Features, and (temp) saving
pm25['noRain9hr'] = ((pm25['RAINFALL_1'] + pm25['RAINFALL_2'] + pm25['RAINFALL_3'] + pm25['RAINFALL_4'] + pm25['RAINFALL_5'] + pm25['RAINFALL_6'] + pm25['RAINFALL_7'] + pm25['RAINFALL_8'] + pm25['RAINFALL_9']) == 0)*1  # NoRain for past 9hr

# 9hr mean, 2th term, 3hr mean, 9hr mean, 
for col in lag_list:
    pm25[col + '_1_p2'] = pm25[col + '_1'] ** 2
    pm25[col + '_mean'] = (pm25[col + '_1'] + pm25[col + '_2'] + pm25[col + '_3'] + pm25[col + '_4'] + pm25[col + '_5'] + pm25[col + '_6'] + pm25[col + '_7'] + pm25[col + '_8'] + pm25[col + '_9']) / 9

# wind x rain
pm25['windrainx'] = pm25['RAINFALL_1'] * pm25['xvec_1']
pm25['windrainy'] = pm25['RAINFALL_1'] * pm25['yvec_1']

# Saving
pm25 = pm25.reset_index()
pm25.to_csv('../output/finalpm25.csv')







## PART 2. Implementing 

## (1) Draw CV
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
    
    return {'fold': fold, 'train':train, 'val':val}  #zip(train, val)





# (2) Gradient Descend for Qudratic Problem 
#    (Note...the standard problem form... argmin 0.5b'Qb + p'b...and Q should be psd)
def GD_Quad(Q, P, x0,step='best',conv_norm=0.1,max_force=100,ada=False):
    
    '''
    For the func form like 0.5(b'Qb) + pb 
    In Quadratic case...there is an optimized stepsize (uniform): inner(gd)/(gd)'Q(gd)
     - If set ada=True...compute "individual learning rate" follow by adaGD
     - conv_norm for convergence norm(gd); max_force for max iteration
     - Caveats...ada+best seems work badly
    '''    
    
    ## Prepare
    Q,P,x0,dim = np.array(Q), np.array(P), np.array(x0), len(x0)
    gd = Q.dot(x0) + P
    force = 0
    gd_sse, conv_log = gd**2, []  # gd_log=[gd1,gd2,...]; conv_log=[norm1,norm2,...]
    
    ## Main
    while np.linalg.norm(gd) >= conv_norm and force <= max_force:
        if force % 1000 == 0:                          # To report progress
            print ('GD current iter',force)
        force += 1

        ada_size = [1/np.sqrt(i) for i in gd_sse]      # To store the "Inverse Root SSE for each dim", updated in each iter
        ada_size = np.array(ada_size)
        
        if step=='best':
            opstep = gd.dot(gd)/(gd.T.dot(Q).dot(gd))  # Scalar            
            if ada == True:
                x0 = x0 - opstep*ada_size*gd           # ada_size is a vector...thus "element-wise product" 
            else:
                x0 = x0 - opstep*gd             
        elif step=='thm':
            x0 = -(np.linalg.inv(Q)).dot(P)        
        elif step > 0:
            if ada == True:
                x0 = x0 - step*ada_size*gd
            else:
                x0 = x0 - step*gd
        
        conv_log.append(np.linalg.norm(gd))
        gd = Q.dot(x0) + P           # update gradient
        gd_sse = gd_sse + gd**2      # update for adaGD
        
    conv_log.append(np.linalg.norm(gd))
    return x0   #[conv_log,x0] #[conv_log,force,x0], [force,x0] x0






# (3) Train The model and report evaluation of each models
#     Take data, cv(draw), and a list of model(In regularied-reg case, a list of lambda)
def Train(feature,y,data,cv,tuning):
    
    '''
    My "Train" is designed as follow: 
    (1) Take data, features, CVdraw, and a tuning list
    (2) Use CVdraw to do CV, report the Eval(and the agg of each folds) of each model
    (3) Select the model with best Eval and train the whole sample upon such model
    (4) Finally, I return (a) Eval report (b) Final Model (c) Other info
    ''' 
    
    folds, models, datasize = cv['fold'], len(tuning), data.shape[0]
    report = np.zeros([models,folds])
    for k in range(folds):
        for m in range(models):
            
            ## Train by training set
            #print ('Current Progress: fold', k, ' and model ', m)
            Xmat = np.array(data.loc[cv['train'][k], feature])
            yvec = np.array(data.loc[cv['train'][k], y])
            b = GD_Quad(2.*(Xmat.T.dot(Xmat) + np.identity(len(feature))*tuning[m]), -2.*Xmat.T.dot(yvec), np.repeat(0.,len(feature)))        # GD args  
            
            ## Evaluate by Validating set
            Xmat = data.loc[cv['val'][k], feature]
            yvec = np.array(data.loc[cv['val'][k], y])
            f1 = lambda x: np.array(x).dot(np.array(b))
            Xpred = np.array(Xmat.apply(f1,axis=1))
            report[m,k] = (Xpred - yvec).T.dot((Xpred - yvec))  # SSE
    
    # Final Model...exploit whole data 
    report_agg = []
    for i in range(models):
        agg_sse = np.sqrt(report[i,].sum() / datasize)
        report_agg.append(agg_sse)
    
    report_agg = np.array(report_agg)
    selected_model = report_agg.argmin()  # The index(model) with min Eval
    print ('CV done, model',selected_model,'selected')
    
    Xmat = np.array(data.loc[:, feature])
    yvec = np.array(data.loc[:, y])
    b = GD_Quad(2.*(Xmat.T.dot(Xmat) + np.identity(len(feature))*tuning[selected_model]), -2.*Xmat.T.dot(yvec), np.repeat(0.,len(feature)),step=0.15,ada=True,max_force=100000)   #  ,ada=True,step=0.15,max_force=5000     
    
    return {'finModel':b,'finFeature':feature,'finModelIndex':selected_model,'report':[report,report_agg]}






## PART 3 ##
pm25 = copypm25
pm25['cept'] = 1

# Mod 3
feature_list = (['cept','CH4_1','CH4_1_p2','CO_1','CO_1_p2','NO_mean','NO2_mean','NOx_1','NOx_mean','O3_1','O3_2','PM10_1','PM2.5_1','noRain9hr','RAINFALL_mean','RAINFALL_1','RAINFALL_2','RAINFALL_3','RAINFALL_4','windrainx','windrainy'] + 
     ['RH_1','SO2_1','SO2_1_p2','THC_1','THC_1_p2','Easthr_1','Southhr_1','Southhr_mean','xvechr_1','xvechr_mean','yvechr_1','xvec_Ehr_1','yvec_Shr_1','East_1','South_1','xvec_1','yvec_1','xvec_E_1','yvec_S_1'])

pm25_list = feature_list + ['PM2.5']
pm25 = pm25[pm25_list]
print (len(feature_list))

cv1 = CVdraw(5624,10)
tune = [0]        #[0,1,5,10] # [1,10,100,1000,10000,100000] ...
y = 'PM2.5'
result = Train(feature_list,y,pm25,cv1,tune)

# Saving
np.save('model_m3fin_copy.npy',result['finModel'])







