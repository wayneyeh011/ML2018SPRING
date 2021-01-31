### Training Revisit 2020 
# (1) Modify process
# (2) Try new model


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import sys



### Processing
# Drop abnormal img
diff=(x_mat.max(axis=1)-x_mat.min(axis=1))
x_mat2=x_mat[np.where(diff>50)[0]]
y_label2=y_label[np.where(diff>50)[0]]
y_mat2=y_mat[np.where(diff>50)[0]]
N2=x_mat2.shape[0]

# Image scaled to [0,255]
diff=(x_mat2.max(axis=1)-x_mat2.min(axis=1))
scaler=(255/diff).reshape(N2,1)
x_mat2=(x_mat2-x_mat2.min(axis=1).reshape(N2,1))*scaler

# data-augment
# featurewise_center, featurewise_std_normalization ... normalize pixel-wise
aug = ImageDataGenerator(rotation_range = 15., 
                         width_shift_range = 0.2, 
                         height_shift_range = 0.1,
                         horizontal_flip = True,
                         brightness_range=(0.5,1.5),
                         shear_range = 0.15,
                         zoom_range=(0.75,1.25),
                         featurewise_center = True,
                         featurewise_std_normalization = True)
aug.fit(x_mat2)



### Model
# Base on ResNet
def call_myRS(deeper=False,param_size=1):
    
    top_input=Input(shape=(48,48,1))
    top_pad=ZeroPadding2D(padding=(1,1))(top_input)
    top_conv=Convolution2D(kernel_size=(3,3),strides=(1,1),filters=32*param_size)(top_pad)
    top_pool=MaxPooling2D(pool_size=(2,2))(top_conv)

    # B1
    B1_bn0=BatchNormalization(axis=-1)(top_pool)
    B1_relu0=Activation('relu')(B1_bn0)
    B1_conv1=Convolution2D(kernel_size=(1,1),filters=32*param_size,padding='same',use_bias=False)(B1_relu0)
    B1_bn1=BatchNormalization(axis=-1)(B1_conv1)
    B1_relu1=Activation('relu')(B1_bn1)
    B1_conv2=Convolution2D(kernel_size=(3,3),filters=32*param_size,padding='same',use_bias=False)(B1_relu1)
    B1_bn2=BatchNormalization(axis=-1)(B1_conv2)
    B1_relu2=Activation('relu')(B1_bn2)
    B1_conv0=Convolution2D(kernel_size=(1,1),filters=128*param_size,padding='same',use_bias=True)(B1_relu0) # *1
    B1_conv3=Convolution2D(kernel_size=(1,1),filters=128*param_size,padding='same',use_bias=True)(B1_relu2) # *2
    B1_add=Add()([B1_conv0,B1_conv3])  # (24,24,128)
    
    # B1'
    B11_bn0=BatchNormalization(axis=-1)(B1_add) # *1
    B11_relu0=Activation('relu')(B11_bn0)
    B11_conv1=Convolution2D(kernel_size=(1,1),filters=32*param_size,padding='same',use_bias=False)(B11_relu0)
    B11_bn1=BatchNormalization(axis=-1)(B11_conv1)
    B11_relu1=Activation('relu')(B11_bn1)
    B11_conv2=Convolution2D(kernel_size=(3,3),filters=32*param_size,padding='same',use_bias=False)(B11_relu1)
    B11_bn2=BatchNormalization(axis=-1)(B11_conv2)
    B11_relu2=Activation('relu')(B11_bn2)
    B11_conv3=Convolution2D(kernel_size=(1,1),filters=128*param_size,padding='same',use_bias=True)(B11_relu2) # *2
    B11_add=Add()([B1_add,B11_conv3])

    # B2
    B2_bn0=BatchNormalization(axis=-1)(B11_add)   
    B2_relu0=Activation('relu')(B2_bn0)
    B2_conv1=Convolution2D(kernel_size=(1,1),filters=32*param_size,padding='same',use_bias=False)(B2_relu0)
    B2_bn1=BatchNormalization(axis=-1)(B2_conv1)
    B2_relu1=Activation('relu')(B2_bn1)
    B2_pad=ZeroPadding2D(padding=(1,1))(B2_relu1)
    B2_conv2=Convolution2D(kernel_size=(3,3),strides=(2,2),filters=32*param_size,use_bias=False)(B2_pad) # shrink (12,12,32)
    B2_bn2=BatchNormalization(axis=-1)(B2_conv2)
    B2_relu2=Activation('relu')(B2_bn2)
    B2_pool=MaxPooling2D(pool_size=(1,1),strides=(2,2))(B1_add) # *1
    B2_conv3=Convolution2D(kernel_size=(1,1),filters=128*param_size,padding='same',use_bias=True)(B2_relu2) # *2
    B2_add=Add()([B2_pool,B2_conv3]) # (12,12,128)


    # B3
    B3_bn0=BatchNormalization(axis=-1)(B2_add)
    B3_relu0=Activation('relu')(B3_bn0)
    B3_conv1=Convolution2D(kernel_size=(1,1),filters=64*param_size,padding='same',use_bias=False)(B3_relu0)
    B3_bn1=BatchNormalization(axis=-1)(B3_conv1)
    B3_relu1=Activation('relu')(B3_bn1)
    B3_conv2=Convolution2D(kernel_size=(3,3),filters=64*param_size,padding='same',use_bias=False)(B3_relu1)
    B3_bn2=BatchNormalization(axis=-1)(B3_conv2)
    B3_relu2=Activation('relu')(B3_bn2)
    B3_conv0=Convolution2D(kernel_size=(1,1),filters=256*param_size,padding='same',use_bias=True)(B3_relu0)
    B3_conv3=Convolution2D(kernel_size=(1,1),filters=256*param_size,padding='same',use_bias=True)(B3_relu2)
    B3_add=Add()([B3_conv0,B3_conv3]) # (12,12,128)
    
    # B3'
    B33_bn0=BatchNormalization(axis=-1)(B3_add) # *1
    B33_relu0=Activation('relu')(B33_bn0)
    B33_conv1=Convolution2D(kernel_size=(1,1),filters=64*param_size,padding='same',use_bias=False)(B33_relu0)
    B33_bn1=BatchNormalization(axis=-1)(B33_conv1)
    B33_relu1=Activation('relu')(B33_bn1)
    B33_conv2=Convolution2D(kernel_size=(3,3),filters=64*param_size,padding='same',use_bias=False)(B33_relu1)
    B33_bn2=BatchNormalization(axis=-1)(B33_conv2)
    B33_relu2=Activation('relu')(B33_bn2)
    B33_conv3=Convolution2D(kernel_size=(1,1),filters=256*param_size,padding='same',use_bias=True)(B33_relu2) # *2
    B33_add=Add()([B3_add,B33_conv3]) # (12,12,128)

    # B4
    B4_bn0=BatchNormalization(axis=-1)(B33_add)
    B4_relu0=Activation('relu')(B4_bn0)
    B4_conv1=Convolution2D(kernel_size=(1,1),filters=64*param_size,padding='same',use_bias=False)(B4_relu0)
    B4_bn1=BatchNormalization(axis=-1)(B4_conv1)
    B4_relu1=Activation('relu')(B4_bn1)
    B4_pad=ZeroPadding2D(padding=(1,1))(B4_relu1)
    B4_conv2=Convolution2D(kernel_size=(3,3),strides=(2,2),filters=64*param_size,padding='valid',use_bias=False)(B4_pad) # shrink (6,6,128)
    B4_bn2=BatchNormalization(axis=-1)(B4_conv2)
    B4_relu2=Activation('relu')(B4_bn2)
    B4_pool=MaxPooling2D(pool_size=(1,1),strides=(2,2))(B3_add) # *1
    B4_conv3=Convolution2D(kernel_size=(1,1),filters=256*param_size,padding='same',use_bias=True)(B4_relu2) # *2
    B4_add=Add()([B4_pool,B4_conv3])

    # Bot
    bot_bn=BatchNormalization(axis=-1)(B4_add)
    bot_relu=Activation('relu')(bot_bn)
    bot_avg=GlobalAveragePooling2D()(bot_relu)
    bot_d=Dense(7,activation='softmax')(bot_avg)

    model = Model(inputs=top_input, outputs=bot_d)
    return model



