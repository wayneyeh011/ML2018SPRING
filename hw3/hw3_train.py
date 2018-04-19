import numpy as np
import pandas as pd
import csv
import sys

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
#import matplotlib.pyplot as plt





### This is the overall training process ###
##  1. Import and prepare (take time)
img = pd.read_csv(sys.argv[1])    ##  '../input/train.csv' sys.arg[]
img['feature'] = img['feature'].str.split(' ')
x_mat = np.array(img['feature'].tolist(), dtype=int).reshape([28709,2304])
y_label = np.array(img['label'], dtype=int)
y_mat = np_utils.to_categorical(y_label,7)
#print (x_mat.shape, y_label.shape , y_mat.shape)


## Normalize  
#x_mat2 = np.zeros([28709, 2304])
#for i in range(2304):
#    x_mat2[:,i] = (x_mat[:,i] - x_mat[:,i].mean()) / (x_mat[:,i].std() + 0.0000001)




## 2. Conv Structure

## Cautious --- input_shape=(48,48,1)
#  ... set an 'input' layers first
# K.set_learning_phase(1)


model = Sequential() 
model.add(InputLayer(input_shape=(48,48,1)))
model.add(Convolution2D(36, (3, 3), activation='relu'))  # , input_shape=(48,48,1)
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

## compiling
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()



## Data augment
aug = ImageDataGenerator(rotation_range = 10., 
                         width_shift_range=0.1, 
                         height_shift_range = 0.1,
                         horizontal_flip = True,
                         shear_range = 0.15)

## check point
#check = ModelCheckpoint('../temp/comp_nor.{epoch:02d}-{val_acc:.2f}.h5', 
#                        monitor='val_acc', save_weights_only=True)  #  save_best_only=True ,


## Train 
val = 25840  # about 90th

#model.fit(x_mat2, y_mat, batch_size=50, epochs=1, validation_split=0.1 ,verbose=1)
model.fit_generator(aug.flow(x_mat2[:val], y_mat[:val], batch_size=32), 
                    steps_per_epoch= val // 32,
                    validation_data = (x_mat2[val:], y_mat[val:]),
                    epochs=50 ,verbose=1, callbacks=[check])


model.save('newtrain.h5')




