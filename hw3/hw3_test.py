import numpy as np
import pandas as pd
import csv
import sys
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint




##
filename = sys.argv[1]       # sys.argv[]  '../input/test.csv'
img = pd.read_csv(filename)    
img['feature'] = img['feature'].str.split(' ')
img = img.sort_values(['id'])
x_test_mat = np.array(img['feature'].tolist(), dtype=int).reshape([7178,2304])
x_test_mat = x_test_mat.reshape([7178,48,48,1])
x_test_mat = x_test_mat / 255




## CNN
model = Sequential() 
model.add(Convolution2D(36, (3, 3), activation='relu', input_shape=(48,48,1)))
model.add(BatchNormalization(axis=-1))
model.add(Convolution2D(36, (3, 3), activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(72, (3, 3), activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Convolution2D(72, (3, 3), activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Convolution2D(72, (3, 3), activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2,2)))

## dense-connected NN
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(400, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(7, activation='softmax'))

##
model.load_weights('pm1.ep75-0.67.h5')  # ../temp/candicate/pm1.ep75-0.67.h5
##

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])




## Prediction
pred = model.predict_classes(x_test_mat, batch_size=100 ,verbose=1)
filename = sys.argv[2]         # sys.argv[] ... '../output/compare'
f1 = open(filename, 'w+')
f1_text = csv.writer(f1,delimiter=',',lineterminator='\n')
f1_text.writerow(['id','label'])
for i in range(x_test_mat.shape[0]):
    f1_text.writerow([int(i), pred[i]])
f1.close()


