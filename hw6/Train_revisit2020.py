### Training Revisit 2020
import numpy as np
import pandas as pd
import sys
import csv
import re

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, InputLayer, Embedding, Dot, Add, Concatenate, Flatten, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Activation



### Models
# sigmoid output, scaled b/t [1,5]
# direct input features (instead of embedded layer)
def call_NEW_m2(user_len, movie_len, emb_dim=20, feature_size=50, deeper=True, linear=True, deeper_k=10, reg=0):
    input_n = Input(shape=[1])
    input_m = Input(shape=[1])
    input_feature = Input(shape=[feature_size])

    # First arg of Embedding (input_dim) is the size of vocabulary
    emb_n = Embedding(user_len, emb_dim,trainable=True,embeddings_initializer='uniform',embeddings_regularizer=regularizers.l1(reg))(input_n)
    emb_n = Flatten()(emb_n)
    emb_m = Embedding(movie_len, emb_dim,trainable=True,embeddings_initializer='uniform',embeddings_regularizer=regularizers.l1(reg))(input_m)
    emb_m = Flatten()(emb_m)
    x = Dot(axes=1)([emb_n,emb_m])
    
    # User/Movie bias
    emb_n_bias = Embedding(user_len, 1,trainable=True,embeddings_initializer='zeros')(input_n)
    emb_n_bias = Flatten()(emb_n_bias)
    emb_m_bias = Embedding(movie_len, 1,trainable=True,embeddings_initializer='zeros')(input_m)
    emb_m_bias = Flatten()(emb_m_bias)
    
    # Features
    if deeper:
        d1=Dense(deeper_k,use_bias=True,activation='relu')(input_feature)
        d1=Dropout(0.5)(d1)
        d1=Dense(1,use_bias=True,activation='relu')(d1)
    else:
        d1=Dense(1,use_bias=True,activation='relu')(input_feature)
    
    # Output
    if linear:
        x = Add()([x, emb_n_bias, emb_m_bias, d1])
    else:
        x = Add()([x, emb_n_bias, emb_m_bias, d1])
        x = Activation('sigmoid')(x)
        x = Lambda(lambda x: x * (5-1) + 1)(x)
    
    model = Model([input_n,input_m,input_feature], x)
    return model


# with 'genres' embedded to a vector (not just a bias)
def call_NEW_m2genemb(user_len, movie_len, emb_dim=20, feature_size=50, 
                      deeper=True, deeper_k=10, reg=0):
    input_n = Input(shape=[1])
    input_m = Input(shape=[1])
    input_feature = Input(shape=[feature_size])
    input_genres = Input(shape=[18]) # length of features 
    
    # First arg of Embedding (input_dim) is the size of vocabulary
    emb_n = Embedding(user_len, emb_dim,trainable=True,embeddings_initializer='uniform',embeddings_regularizer=regularizers.l1(reg))(input_n)
    emb_n = Flatten()(emb_n)
    emb_m = Embedding(movie_len, emb_dim,trainable=True,embeddings_initializer='uniform',embeddings_regularizer=regularizers.l1(reg))(input_m)
    emb_m = Flatten()(emb_m)
    x = Dot(axes=1)([emb_n,emb_m])
    
    # User/Movie bias
    emb_n_bias = Embedding(user_len, 1,trainable=True,embeddings_initializer='zeros')(input_n)
    emb_n_bias = Flatten()(emb_n_bias)
    emb_m_bias = Embedding(movie_len, 1,trainable=True,embeddings_initializer='zeros')(input_m)
    emb_m_bias = Flatten()(emb_m_bias)
    
    # genres-embedded
    genres_emb = Dense(emb_dim,use_bias=False)(input_genres)
    x2 = Dot(axes=1)([emb_n,genres_emb])
    
    # Features
    if deeper:
        d1=Dense(deeper_k,use_bias=True,activation='relu')(input_feature)
        d1=Dropout(0.5)(d1)
        d1=Dense(1,use_bias=True,activation='relu')(d1)
    else:
        d1=Dense(1,use_bias=True,activation='relu')(input_feature)
    
    # Output
    x = Add()([x, x2, emb_n_bias, emb_m_bias, d1])
    x = Activation('sigmoid')(x)
    x = Lambda(lambda x: x * (5-1) + 1)(x)
    
    model = Model([input_n,input_m,input_feature,input_genres], x)
    return model






## Based on ordered logit
from tensorflow.keras.layers import Layer
class BiasLayer(Layer):
    def __init__(self, units, *args, **kwargs):
        super(BiasLayer, self).__init__(*args, **kwargs)
        self.bias = self.add_weight('bias',
                                    shape=[units],
                                    initializer='zeros',
                                    trainable=True)
    def call(self, x):
        return -x +self.bias
    
w1=np.array([[1,0,0,0],
             [-1,1,0,0],
             [0,-1,1,0],
             [0,0,-1,1],
             [0,0,0,-1]])
w1=w1.T
b1=np.array([0,0,0,0,1])

def call_NEW_m3(user_len, movie_len, emb_dim=20, feature_size=50, deeper=True, deeper_k=10):
    input_n = Input(shape=[1])
    input_m = Input(shape=[1])
    input_feature = Input(shape=[feature_size])

    # First arg of Embedding (input_dim) is the size of vocabulary
    emb_n = Embedding(user_len, emb_dim,trainable=True,embeddings_initializer='uniform')(input_n)
    emb_n = Flatten()(emb_n)
    emb_m = Embedding(movie_len, emb_dim,trainable=True,embeddings_initializer='uniform')(input_m)
    emb_m = Flatten()(emb_m)
    x = Dot(axes=1)([emb_n,emb_m])
    
    # User/Movie bias
    emb_n_bias = Embedding(user_len, 1,trainable=True,embeddings_initializer='zeros')(input_n)
    emb_n_bias = Flatten()(emb_n_bias)
    emb_m_bias = Embedding(movie_len, 1,trainable=True,embeddings_initializer='zeros')(input_m)
    emb_m_bias = Flatten()(emb_m_bias)
    
    # Features
    if deeper:
        d1=Dense(deeper_k,use_bias=True,activation='relu')(input_feature)
        d1=Dropout(0.5)(d1)
        d1=Dense(1,use_bias=True,activation='relu')(d1)
    else:
        d1=Dense(1,use_bias=True,activation='relu')(input_feature)
    
    x = Add()([x, emb_n_bias, emb_m_bias, d1])
    x = BiasLayer(4,weights=np.array([[0.2,0.4,0.6,0.8]]))(x)
    x = Activation('sigmoid')(x)
    x = Dense(5,trainable=False,weights=[w1,b1])(x)        
    
    model = Model([input_n,input_m,input_feature], x)
    return model


