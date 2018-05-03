import numpy as np
import pandas as pd
import csv
import sys
from time import time

from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, InputLayer, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn import cluster
from sklearn import manifold
from sklearn.decomposition import PCA
from skimage import filters




## HW4 Image Clustering 

## Note 
# 1. argv[1] - argv[3] 
# 2. dir to import .h5
# 3. Setting like # of PCs, whiten, ...



## 1. Load and preprocess
img = np.load(sys.argv[1])   # sys.argv[1]   '../input/image.npy'

# filter noise
t0 = time()
img2 = img.reshape([140000,28,28])
for i in range(140000):
    
    if i % 10000 == 0:
        print ('De_noise at pic', i)        
    cond = ((img2[i,:,:] == 255) | (img2[i,:,:] ==0))
    img2[i,:,:] = filters.median(img2[i,:,:])*cond + img2[i,:,:]*(~cond)
print ('consume: ', time()-t0)




## 2-1 CNN AutoEncoder Load model 
img2 = img2.reshape([140000, 28, 28, 1]) / 255

CAE = Sequential()
CAE.add(InputLayer(input_shape=(28,28,1)))
CAE.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))
CAE.add(MaxPooling2D(pool_size=(2,2), padding='same'))
CAE.add(Convolution2D(16, (3, 3), activation='relu', padding='same'))
CAE.add(MaxPooling2D(pool_size=(2,2), padding='same'))
CAE.add(Convolution2D(8, (3, 3), activation='relu', padding='same'))
CAE.add(MaxPooling2D(pool_size=(2,2), padding='same'))

CAE.add(Convolution2D(8, (3, 3), activation='relu', padding='same'))
CAE.add(UpSampling2D((2,2)))
CAE.add(Convolution2D(16, (3, 3), activation='relu', padding='same'))
CAE.add(UpSampling2D((2,2)))
CAE.add(Convolution2D(32, (3, 3), activation='relu'))
CAE.add(UpSampling2D((2,2)))
CAE.add(Convolution2D(1, (3, 3), activation='sigmoid', padding='same'))

##
CAE.load_weights('CAE.53-0.19.h5')   # '../temp/all/CAE.53-0.19.h5' 'CAE.53-0.19.h5'
##

CAE.compile(loss='binary_crossentropy',
            optimizer='adadelta')
CAE.summary()



## 2-2 CAE_encode
encoder = Model(inputs=CAE.input, outputs=CAE.layers[6].output)
encoder_output = encoder.predict(img2, verbose=1)
print (encoder_output.shape)

encoder_output = encoder_output.reshape([140000, 128])
print (encoder_output.shape)




## 3-1 PCA 
## White or not ? How many PCs
cae_pca = PCA(24, whiten=True)  # 32? 
cae_pca = cae_pca.fit(encoder_output)
cae_pca_output = cae_pca.fit_transform(encoder_output)
print (cae_pca_output.shape)
print ((cae_pca.explained_variance_ratio_).cumsum())




## 3-2 Normalize PCs
cae_pca_output2 = np.zeros([140000, 24])  #40
for i in range(24):
    cae_pca_output2[:,i] = (cae_pca_output[:,i] - cae_pca_output[:,i].mean()) / (cae_pca_output[:,i].std() + 0.0001)



## 3-3 Kmeans
data = cae_pca_output2 #encoder_output2 #encoder_output  #cae_pca_output  #encoder_output2
clabel_times = {}
interia_collect = []
for i in range(1,11):
    
    print ('kmean times=', i, 'start')
    all_km = cluster.KMeans(2, verbose=0)  # Maybe...not stick to 10 cluster
    t0 = time()
    all_km.fit(data)
    clabel_times[str(i)] = all_km.labels_
    interia_collect.append(all_km.inertia_)
    print ('kmean times=', i, 'Done, time consumption=', time()-t0, 'interia=', all_km.inertia_)




## 3-4 Take a look
#import matplotlib.pylab as plt
#
#clu = np.where(clabel_times['1'] == 0)[0]
#start = 0
#plt.figure(figsize=(15, 4))
#for i in range(50):
#    # display original
#    ax = plt.subplot(5, 10, i + 1)   # 2x10 num.1
#    plt.imshow(img2[clu[i]].reshape(28, 28))
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#plt.show()




## 3-5 Predicting 
t0 = time()
test_case = pd.read_csv(sys.argv[2])  # sys.argv[2]  '../input/test_case.csv'  
test_case = test_case.sort_values(['ID'])
test_case = np.array(test_case)


times_arg = np.array(interia_collect).argmin() + 1  ## Note...times {1, ... } that is index + 1
clabel = clabel_times[str(times_arg)]  ## Can be other like...all_km.labels_


filename = sys.argv[3]   # sys.argv[]  '../output/cae_pcnor_km'
f1 = open(filename, 'w+')
f1_text = csv.writer(f1,delimiter=',',lineterminator='\n')
f1_text.writerow(['ID','Ans'])
for i in range(test_case.shape[0]):
    id1, id2 = test_case[i,1], test_case[i,2]
    ans = (clabel[id1] == clabel[id2])*1
    f1_text.writerow([int(i), ans]) # pred[i]
f1.close()
print ('predict done, consume time:', time()-t0)









