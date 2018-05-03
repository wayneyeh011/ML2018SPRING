import numpy as np
import pandas as pd
import csv
import sys
from time import time





## 0. Read Me
#  (1) Using model from hw1
#  (2) Bagging is bult on top of regression
#  (3) Without lossing of purpose, I use theoretical (analytical) solution in regression
#  (4) I use a preprocessed dataset produced by myself previously
#  (5) Bagging do not improve --- probably, I have a coding error or it's indeed bagging do not help in this case ??? 
#  ...




## 1. Gradient Descend 
def GD_Quad(Q, P, x0, step='best',conv_norm=0.1,max_force=100,ada=False):
    
    '''
    For the func form like 0.5(b'Qb) + pb 
    In Quadratic case...there is an optimized stepsize (uniform): inner(gd)/(gd)'Q(gd)
     - If set ada=True...use "individual learning rate" follow by adaGD
     - conv_norm for convergence norm(gd), max_force for max iteration
     - Caveats...ada+best work badly
    '''    
    
    ## Prepare
    Q,P,x0,dim = np.array(Q), np.array(P), np.array(x0), len(x0)
    gd = Q.dot(x0) + P
    force = 0
    gd_sse, conv_log = gd**2, []  # gd_log=[gd1,gd2,...]; conv_log=[norm1,norm2,...]
    
    ## Main
    while np.linalg.norm(gd) >= conv_norm and force <= max_force:
        if force % 1000 == 0:
            print ('GD current iter',force)
        force += 1

        ada_size = [1/np.sqrt(i) for i in gd_sse]    # To store the "Inverse Root SSE for each dim", updated in each iter
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
        gd = Q.dot(x0) + P        # update
        gd_sse = gd_sse + gd**2
        #print (ada_size,gd_sse,gd)
        
    conv_log.append(np.linalg.norm(gd))
    return x0  






## 1. Importing, data from analysis
pm25 = pd.read_csv('../output/finalpm25.csv')  # preprocessed data by myself previously
pm25['cept'] = 1
f2_list = []
for col in ['AMB_TEMP','CH4','CO','NMHC','NO','NO2','NOx','O3','PM10','PM2.5','RAINFALL','RH','SO2','THC','xvechr','yvechr','xvec','yvec']:
    f2_list += [col + '_' + str(i) for i in range(1,10)]
feature_list = ['cept'] + f2_list

pm25_list = feature_list + ['PM2.5']
pm25 = pm25[pm25_list]
print (len(feature_list), pm25.shape)




# 2.  Cut Validating set
num_of_train = 5000
val = np.random.permutation(pm25.shape[0])[num_of_train:]
tr = np.array(list(set(np.arange(pm25.shape[0])) - set(val)))
print ('val examples:', val[:5])
pm25_train = pm25.loc[tr, :]
pm25_val = pm25.loc[val, :]
print ('Train shape',pm25_train.shape, 'Val shape', pm25_val.shape)

Xmat = np.array(pm25_train.loc[:,feature_list])
yvec = np.array(pm25_train.loc[:,'PM2.5'])
val_xmat = np.array(pm25_val.loc[:,feature_list])
val_yvec = np.array(pm25_val.loc[:,'PM2.5'])
print (Xmat.shape, yvec.shape,val_xmat.shape,val_yvec.shape)





# 3. Bagging
bag_times = 150 
size, dim = Xmat.shape[0], Xmat.shape[1]
w_log = np.zeros([bag_times,dim])   # (bs_times x K)

for b in range(bag_times):
    
    ## BS data
    print ('----------','bs times',b+1)
    bs = np.random.randint(0,size,size)   #bs = np.arange(size)
    print ('First 5 BS index',bs[:5], 'last 5 BS index', bs[-5:])
    
    ## Training
    Xmat2 = Xmat[bs,:]
    yvec2 = yvec[bs]
    Qmat = 2.*(Xmat2.T.dot(Xmat2))   # (k x k)
    pvec = -2.*Xmat2.T.dot(yvec2)    # (k x 1)
    w0 = np.repeat(0., dim)
    w = GD_Quad(Qmat, pvec, w0, step='thm') 
    print ('GD done')
    w_log[b,:] = w.flatten() 
    print ('Save in coef collector')
    
    ## Bagging accessing
    Xhat_mat = Xmat2.dot((w_log[:b+1,:]).T)
    Xhat = np.apply_along_axis(np.mean, 1, Xhat_mat)
    SSR = ((yvec2 - Xhat)**2).sum()
    RMSE = np.sqrt(SSR / size)
    print ('bs',b+1,'done', 'bs_RMSE', RMSE, '\n')





## 4. Assessing On Validing Set (Probably a smarter approach is Out-Of-Bag??) 
num_of_bs = 150    # First ? bagging results are used (and thus averaged)
w_log2 = w_log[:num_of_bs,:]
val_hatmat = val_xmat.dot(w_log2.T)
val_hat = np.apply_along_axis(np.mean,1,val_hatmat)
SSR = ((val_yvec - val_hat)**2).sum()
RMSE = np.sqrt(SSR / val_xmat.shape[0])
print ('Val RMSE', RMSE)















