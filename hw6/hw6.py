import numpy as np
import pandas as pd
import sys
import csv
from time import time

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, InputLayer, Embedding, Dot, Add, Concatenate, Flatten, Lambda
from keras.optimizers import Adam
from keras.constraints import min_max_norm
from keras import regularizers
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint




## 0.

#rating_path
test_path = sys.argv[1]  # '../input/test.csv'
user_path = sys.argv[4]  # '../input/users.csv'
movie_path = sys.argv[3]  # '../input/movies.csv'
model_path = 'm2_185_bias.05-0.7228.h5' # '../temp/MF/m2_trial_5_largerbs_search/m2_185_bias.05-0.7228.h5', 'm2_185_bias.05-0.7228.h5'
output_path = sys.argv[2] # 'mf_m2'

# Models
# m2 (mf model) 



## 1 Read and Tokenize

# Read
user = pd.read_csv(user_path, delimiter='::', engine='python')  # '../input/users.csv'
movie = pd.read_csv(movie_path, delimiter='::', engine='python')  # '../input/movies.csv'
movie['Genres'] = movie['Genres'].str.split('|')


# User/Movie Token
user_list = user['UserID'].unique()
user_list.sort()
user_mat = pd.DataFrame({'UserID':user_list, 'UserToken':np.arange(len(user_list))})
user = pd.merge(user, user_mat, left_on='UserID', right_on='UserID', how='left')
print ('USER',len(user_list), user_list.min(), user_list.max())

movie_list = movie['movieID'].unique()
movie_list.sort()
movie_mat = pd.DataFrame({'MovieID':movie_list, 'MovieToken':np.arange(len(movie_list))})
movie = pd.merge(movie, movie_mat, left_on='movieID', right_on='MovieID', how='left')
print ('MOVIE',len(movie_list), movie_list.min(), movie_list.max())


# More User
u_list = user['Gender'].unique()
u_list.sort()
user_gender_len = len(u_list)
u_mat = pd.DataFrame({'Gender':u_list, 'GenderToken':np.arange(len(u_list))})
user = pd.merge(user, u_mat, left_on='Gender', right_on='Gender', how='left')

u_list = user['Age'].unique()
u_list.sort()
user_age_len = len(u_list)
u_mat = pd.DataFrame({'Age':u_list, 'AgeToken':np.arange(len(u_list))})
user = pd.merge(user, u_mat, left_on='Age', right_on='Age', how='left')

u_list = user['Occupation'].unique()
u_list.sort()
user_occu_len = len(u_list)
u_mat = pd.DataFrame({'Occupation':u_list, 'OccupationToken':np.arange(len(u_list))})
user = pd.merge(user, u_mat, left_on='Occupation', right_on='Occupation', how='left')
print ('The length of unique Gender, Age, Occupation:', user_gender_len, user_age_len, user_occu_len)


# More Movie
movie['Era'] = movie['Title'].str.split('(')
movie['Era'] = movie['Era'].apply(lambda x: int(x[-1][:-1]))
movie.loc[movie['Era'] <= 1930, 'Era'] = 1930
movie.loc[((movie['Era'] > 1930) & (movie['Era'] <= 1940)), 'Era'] = 1940
movie.loc[((movie['Era'] > 1940) & (movie['Era'] <= 1950)), 'Era'] = 1950
movie.loc[((movie['Era'] > 1950) & (movie['Era'] <= 1960)), 'Era'] = 1960
movie.loc[((movie['Era'] > 1960) & (movie['Era'] <= 1970)), 'Era'] = 1970
movie.loc[((movie['Era'] > 1970) & (movie['Era'] <= 1980)), 'Era'] = 1980
m_list = movie['Era'].unique()
m_list.sort()
movie_era_len = len(movie['Era'].unique())
m_mat = pd.DataFrame({'Era':m_list, 'EraToken':np.arange(len(m_list))})
movie = pd.merge(movie, m_mat, left_on='Era', right_on='Era', how='left')


m_list = []
for i in range(3883):
    m_list = m_list + movie['Genres'][i]    
m_list = pd.Series(m_list).unique()
m_list.sort()

GenresToken_dict = {}
for i in range(len(m_list)):
    GenresToken_dict[m_list[i]] = i
movie_genres_len = len(m_list)
movie['GenresToken'] = movie['Genres'].apply(lambda x: [GenresToken_dict[i] for i in x])

def pad_genres(x):
    'In this case...max length is 6'     ## NOTE ... One should check the max length...
    offset = 6 - len(x)
    return [18]*offset + x               ## The length of token dict + 1 (the token of empty)
movie['GenresToken'] = movie['GenresToken'].apply(pad_genres)
movie_genres_len = movie_genres_len + 1  ## Add the token of empty


print ('The length of unique Era, Genres:', movie_era_len, movie_genres_len)





## 2 Merging and Get Token

user = user[['UserID','UserToken','GenderToken','AgeToken','OccupationToken']]
movie = movie[['MovieID','MovieToken','EraToken','GenresToken']]

testing = pd.read_csv(test_path)  # '../input/test.csv'
testing = testing.sort_values('TestDataID')
testing = pd.merge(testing, user, left_on='UserID', right_on='UserID', how='left')
testing = pd.merge(testing, movie, left_on='MovieID', right_on='MovieID', how='left')
print (testing.shape)
print (testing.head())

# get id token
test_user_token = np.array(testing['UserToken'])
test_movie_token = np.array(testing['MovieToken'])
print (len(test_user_token), len(test_movie_token))
print (test_user_token.min(), test_user_token.max())
print (test_movie_token.min(), test_movie_token.max())

# get other token
test_gender_token = np.array(testing['GenderToken'])
test_age_token = np.array(testing['AgeToken'])
test_occu_token = np.array(testing['OccupationToken'])
test_era_token = np.array(testing['EraToken'])
test_genres_token = []
test_genres_s = testing['GenresToken']

for i in range(len(testing)):
    test_genres_token.append(test_genres_s[i])
    
test_genres_token = np.array(test_genres_token)
print (test_genres_token.shape)




## 3. Get Model

n_len = len(user_list)
m_len = len(movie_list)
print ('Length of user and movie ID:', n_len, m_len)


def oper(x):
    return K.sum(x, axis=1, keepdims=True)

def call_m2(emb_dim=20,reg=False,reg_r=0.01):
    input_n = Input(shape=[1])
    input_m = Input(shape=[1])

    input_gender = Input(shape=[1])
    input_age = Input(shape=[1])
    input_occup = Input(shape=[1])
    input_era = Input(shape=[1])
    input_genres = Input(shape=[6])

    # First arg (input_dim) is the size of vocabulary
    if reg:
        emb_n = Embedding(n_len, emb_dim,trainable=True,embeddings_initializer='uniform',embeddings_regularizer=regularizers.l2(reg_r))(input_n)
        emb_n = Flatten()(emb_n)
        emb_m = Embedding(m_len, emb_dim,trainable=True,embeddings_initializer='uniform',embeddings_regularizer=regularizers.l2(reg_r))(input_m)
        emb_m = Flatten()(emb_m)
        x = Dot(axes=1)([emb_n,emb_m])
    else:
        emb_n = Embedding(n_len, emb_dim,trainable=True,embeddings_initializer='uniform')(input_n)
        emb_n = Flatten()(emb_n)
        emb_m = Embedding(m_len, emb_dim,trainable=True,embeddings_initializer='uniform')(input_m)
        emb_m = Flatten()(emb_m)
        x = Dot(axes=1)([emb_n,emb_m])        
    
    emb_n_bias = Embedding(n_len, 1,trainable=True,embeddings_initializer='zeros')(input_n)
    emb_n_bias = Flatten()(emb_n_bias)
    emb_m_bias = Embedding(m_len, 1,trainable=True,embeddings_initializer='zeros')(input_m)
    emb_m_bias = Flatten()(emb_m_bias)

    emb_gender = Embedding(user_gender_len, 1,trainable=True,embeddings_initializer='zeros')(input_gender)
    emb_gender = Flatten()(emb_gender)
    emb_age = Embedding(user_age_len, 1,trainable=True,embeddings_initializer='zeros')(input_age)
    emb_age = Flatten()(emb_age)
    emb_occup = Embedding(user_occu_len, 1,trainable=True,embeddings_initializer='zeros')(input_occup)
    emb_occup = Flatten()(emb_occup)

    emb_era = Embedding(movie_era_len, 1,trainable=True,embeddings_initializer='zeros')(input_era)
    emb_era = Flatten()(emb_era)
    emb_genres = Embedding(movie_genres_len, 1,trainable=True,embeddings_initializer='zeros')(input_genres)
    emb_genres = Flatten()(emb_genres)
    emb_genres = Lambda(oper)(emb_genres)

    x = Add()([x, emb_n_bias, emb_m_bias, emb_gender, emb_age, emb_occup, emb_era, emb_genres])
    model = Model([input_n,input_m,input_gender,input_age,input_occup,input_era, input_genres], x)
    model.compile(loss='mse', optimizer='adam')  #...  , metrics=['accuracy']
    #model.summary()
    return model


def call_m25(emb_dim=20):
    input_n = Input(shape=[1])
    input_m = Input(shape=[1])

    input_gender = Input(shape=[1])
    input_age = Input(shape=[1])
    input_occup = Input(shape=[1])
    input_era = Input(shape=[1])
    input_genres = Input(shape=[6])


    # First arg (input_dim) is the size of vocabulary
    emb_n = Embedding(n_len, emb_dim,trainable=True,embeddings_initializer='uniform')(input_n)
    emb_n = Flatten()(emb_n)
    emb_m = Embedding(m_len, emb_dim,trainable=True,embeddings_initializer='uniform')(input_m)
    emb_m = Flatten()(emb_m)
    x = Dot(axes=1)([emb_n,emb_m])        
    
    emb_gender = Embedding(user_gender_len, 1,trainable=True,embeddings_initializer='zeros')(input_gender)
    emb_gender = Flatten()(emb_gender)
    emb_age = Embedding(user_age_len, 1,trainable=True,embeddings_initializer='zeros')(input_age)
    emb_age = Flatten()(emb_age)
    emb_occup = Embedding(user_occu_len, 1,trainable=True,embeddings_initializer='zeros')(input_occup)
    emb_occup = Flatten()(emb_occup)

    emb_era = Embedding(movie_era_len, 1,trainable=True,embeddings_initializer='zeros')(input_era)
    emb_era = Flatten()(emb_era)
    emb_genres = Embedding(movie_genres_len, 1,trainable=True,embeddings_initializer='zeros')(input_genres)
    emb_genres = Flatten()(emb_genres)
    emb_genres = Lambda(oper)(emb_genres)

    x = Add()([x, emb_gender, emb_age, emb_occup, emb_era, emb_genres])
    model = Model([input_n,input_m,input_gender,input_age,input_occup,input_era, input_genres], x)
    model.compile(loss='mse', optimizer='adam')  #...  , metrics=['accuracy']
    #model.summary()
    return model




model = call_m2(emb_dim=185,reg=False)
# model = call_m25(emb_dim=185)
model.load_weights(model_path)   # '../temp/MF/m2_trial_5_largerbs_search/m2_185_bias.05-0.7228.h5'





## 4. Output
pred = model.predict([test_user_token,test_movie_token, test_gender_token, test_age_token, test_occu_token, test_era_token, test_genres_token], verbose=1)
pred = pred.squeeze()

# (denormalization...if needed...need to ofter rate.std() and rate.mean()) 
#pred = pred*rate.std() + rate.mean()


# Top-Floor
pred[pred > 5] = 5
pred[pred < 1] = 1
print ('First 10 pred:', pred[:10])


# Output
testID = testing['TestDataID']

size = len(pred)
t0 = time()
filename = output_path
f1 = open(filename, 'w+')
f1_text = csv.writer(f1,delimiter=',',lineterminator='\n')
f1_text.writerow(['TestDataID','Rating'])
for i in range(size):
    f1_text.writerow([int(testID[i]), pred[i]]) # pred[i]
f1.close()
print ('Time consumption:', time()-t0)






