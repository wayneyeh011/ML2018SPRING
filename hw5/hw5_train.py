import numpy as np
import pandas as pd
import csv
import sys
from time import time

from keras.models import Sequential, Model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, InputLayer, Embedding
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

from gensim.models import Word2Vec, KeyedVectors



## The structure is organized as follow...

# (1) Train Word2Vec by gensim ---- exploit label and unlabel data
# (2) RNN structure
# [Semi-Supervise training is noted in another .py (Semi.py)]



## Arguments and load file ---
lab_train_path = sys.argv[1]        #sys.argv[1]  '../input/training_label.txt'
unlab_train_path = sys.argv[2]      #sys.argv[2]  '../input/training_nolabel.txt'
mywv_path = 'mywv'  # '../temp/w2v_TO_cbow100-5-3_proc_punc_wv'
call_back_path = 'cb.{epoch:02d}-{val_acc:.3f}.h5'  # '../temp/RNN1/cb100-5-3_TO_pr1_punc_m2.{epoch:02d}-{val_acc:.3f}.h5'



## (1) Train Word2Vec by gensim

#  Read data
f = open(lab_train_path)    ###
X, Y = [], []
for line in f:
    line_list = line.strip().split(' +++$+++ ')
    X.append(line_list[1])
    Y.append(line_list[0])
f.close()
print ('length of labelled texts:',len(X))

f = open(unlab_train_path)  ###
unlab_X = []
for line in f:
    line_list = line.strip()
    unlab_X.append(line_list)
f.close()
print ('length of un-labelled texts:',len(unlab_X))




# Text word sequence and adress ' (Say [can ' t] -> [cant])

bigX = X + unlab_X
X_wseq = []
for i in range(len(bigX)):
    wseq = text_to_word_sequence(bigX[i], filters='"#$%&()*+-/<=>@[\\]^_`{|}~\t\n')  # Keep ! ? . , : ;
    
    if i % 100000 == 0:
        print ('at text', i)
    
    while "'" in wseq:
        abbr = wseq.index("'")
        if ((abbr > 0) & (abbr + 1 < len(wseq))):
            wseq[abbr-1] = wseq[abbr-1] + wseq[abbr+1]
            wseq.pop(abbr)
            wseq.pop(abbr)
        elif abbr == 0:
            wseq.pop(abbr)
        elif abbr + 1 == len(wseq):
            wseq.pop(abbr)

    X_wseq.append(wseq)
print (len(X_wseq))


# W2V Build and Training and Saving
model = Word2Vec(X_wseq, sg=0, size=100, window=5, min_count=3)
print ('size of vocab-dict',len(model.wv.vocab))
print ('Num of texts',model.corpus_count)

model.train(X_wseq, total_examples= len(X_wseq), epochs=15)
model.wv.save(mywv_path)  ###







## (2) RNN

#  Load embedding model
mywv = KeyedVectors.load(mywv_path)  ###
vocab_list = mywv.index2word
print ('vocab size', len(vocab_list))


#  Setting
size = len(X)
max_length = 40 # Dist of length...?
emb_dim = 100
print ('Data size, length of a text, embedded dim:', size, max_length, emb_dim)

# Wseq


# Index of list of text with words not in vocab_list
t0 = time()
proc_infreq_list = []
for text_i in range(size):  # Smaller size for trial
    
    ## Monotor progress
    if (text_i % 20000 == 0):
        print ('At text:', text_i)
        print ('Now Consumption',time()-t0)    
        
    ## Filter out infreq words
    for word in X_wseq[text_i]:
        if not word in vocab_list:
            proc_infreq_list.append(text_i)
            break
            
print ('Time Consumption:', time()-t0)
print ('Length of infrequent word:', len(proc_infreq_list))



# Excluding words not in vocab_list and padding to (40,100)
t0 = time()
print ('Num of text needed to be preprocess:', len(proc_infreq_list))
for text_i in proc_infreq_list:
        
    ## Filter out infreq words
    X_wseq[text_i] = [ele for ele in X_wseq[text_i] if ele in vocab_list]  ## just eliminate or replace with a token
    #X_wseq[text_i] = [ele if ele in vocab_list else 'InFreq_Token' for ele in X_wseq[text_i]]

print ('Time Consumption:', time()-t0)


X_train = np.zeros([size, max_length, emb_dim])
t0 = time()
for text_i in range(size):    
    
    ## Out of 40 and 0 (In some case...filter out infreq words make text empty)
    if len(X_wseq[text_i]) == 0:
        pad_emb_text = np.zeros([max_length, emb_dim])
        X_train[text_i,:,:] = pad_emb_text
    else:    
        emb_text = mywv[X_wseq[text_i]]
        num_pad = 40 - len(X_wseq[text_i])
        pad_emb_text = np.pad(emb_text,pad_width=((num_pad,0),(0,0)), mode='constant')
        X_train[text_i,:,:] = pad_emb_text
        
print ('Time Consumption:', time()-t0)



# RNN model
Y_train = to_categorical(Y)
print (X_train.shape)
print (Y_train.shape)


print ('Traing Model')
model = Sequential()
model.add(LSTM(200, input_shape=(40,100), return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(100, return_sequences=False))

model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
# model.load_weights()


##
print ('Traing Start, Cross my finger and hope everythings goes well')
check = ModelCheckpoint(call_back_path, monitor='val_acc', save_weights_only=True, save_best_only=True)  ###

val = 180000
ep = 7
model.fit(X_train[:val,:,:], Y_train[:val,:], verbose=1, epochs=ep, batch_size= 200, 
          callbacks=[check], validation_data = (X_train[val:,:,:], Y_train[val:,:]))

print ('Training Done, Cheer')
model.model.save_weights('rnn_model')  ###











