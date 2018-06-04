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




## Semi - Supervise Training

# (1) Preprocess Data --- take VERY LONG TIME, because unlab data is HUGE 
#     - I have saved a preprocessed txt, ... which could save time
#     - I will assume the use of such preprocessed txt file
# (2) Setting and Semi-Supervise Training 





## (1) Load Preprocessed data

f = open('../temp/semi/proctxt_lab')
X_wseq = []
for line in f:
    line_list = line.strip().split(' ')
    X_wseq.append(line_list)
f.close()
print ('length of labelled texts:',len(X_wseq))

f = open('../temp/semi/proctxt_unlab')
unlab_X_wseq = []
for line in f:
    line_list = line.strip().split(' ')
    unlab_X_wseq.append(line_list)
f.close()
print ('length of un-labelled texts:',len(unlab_X_wseq))

f = open('../input/training_label.txt')
Y = []
for line in f:
    line_list = line.strip().split(' +++$+++ ')
    Y.append(line_list[0])
f.close()
print ('length of labelled texts:',len(Y))



# Model 
mywv = KeyedVectors.load('../temp/w2v_TO_cbow100-5-3_proc_punc_wv')
vocab_list = mywv.index2word
print ('vocab size', len(vocab_list))


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

##
model.load_weights('../temp/RNN1/cb100-5-3_TO_pr1_punc_m2.04-0.828.h5')
##






## (2) Setting 

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, mywv, batch_size=200, max_length=40, emb_dim=100, n_classes=2, shuffle=True, pred=False): 
        'Initialization'
        self.max_length = max_length
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.pred = pred
        self.mywv = mywv

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # Find list of IDs
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        return self.__data_generation(list_IDs_temp)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'   # X : (n_samples, *dim, n_channels)
                    
        X = np.empty((self.batch_size, self.max_length, self.emb_dim))   # (200, 100, 40)
        
        for ID,text_ID in enumerate(list_IDs_temp):  #  Something like ... [1,3,8,10] ?

            ## Out of 40, 0, less than 40
            if ((len(big_wseq[text_ID]) == 1) & (big_wseq[text_ID][0] == '')):
                pad_emb_text = np.zeros([self.max_length, self.emb_dim])
                X[ID,:,:] = pad_emb_text
            elif len(big_wseq[text_ID]) >= 40:
                emb_text = self.mywv[big_wseq[text_ID][:40]]
                X[ID,:,:] = emb_text
            else:    
                emb_text = self.mywv[big_wseq[text_ID]]
                num_pad = 40 - len(big_wseq[text_ID])
                pad_emb_text = np.pad(emb_text, pad_width=((num_pad,0),(0,0)), mode='constant')
                #print (X[ID,:,:].shape, pad_emb_text.shape)
                X[ID,:,:] = pad_emb_text

        if self.pred:
            return X
        else:
            y = np.empty((self.batch_size), dtype=int)
            for ID,text_ID in enumerate(list_IDs_temp):
                
                y[ID] = self.labels[text_ID]
            return X, to_categorical(y, num_classes=self.n_classes)



def call_m2():
    m2 = Sequential()
    m2.add(LSTM(200, input_shape=(40,100), return_sequences=True))
    m2.add(Dropout(0.4))
    m2.add(LSTM(100, return_sequences=False))

    m2.add(Dense(100, activation='relu'))
    m2.add(Dropout(0.2))
    m2.add(Dense(20, activation='relu'))
    m2.add(Dense(2, activation='softmax'))
    m2.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return m2



## ID_LIST and ID-LABEL DICT (for datagenerator)
big_wseq = X_wseq + unlab_X_wseq
print (len(big_wseq))

ind_log = {}
ind_log['Train'] = list(np.arange(180000))          #list(np.arange(180000))
ind_log['Val'] = list(np.arange(180000,200000))     #list(np.arange(180000,200000))
ind_log['Semi'] = list(np.arange(200000,1378614))   #list(np.arange(200000,1378614))

ind_label = dict(zip(np.arange(200000), Y))
print (ind_log['Train'][:5], ind_log['Val'][:5], ind_log['Semi'][:5])



## Semi-Training

# To have a toy test ... feed list_IDs = ind_log['Semi'][:1000] and steps= len(ind_log['Semi'][:1000] ... (fit_gen is similar)
# retrain + more epoch (4-5 ?)
# not retrain + less epoch (2-3 ?)

th = 0.87
semi_batch = 400000
semi_iter = 1
retrain = True
train_ep = 5


for semi_times in range(semi_iter):
        
    ## (1) Predicting
    print ('\n','Semi_train, Times:', semi_times + 1, '---------------')
    print ('Current size of Semi:', len(ind_log['Semi']))
    test_g = DataGenerator(list_IDs = ind_log['Semi'][:semi_batch], labels = None, 
                           mywv=mywv, pred=True, shuffle=False)
    pred = model.predict_generator(test_g, steps= len(ind_log['Semi'][:semi_batch]) // 200, verbose=1)
    print ('test_pred_shape:', pred.shape)
    
           
    ## (2) Evaluating
    pick_num = ((pred[:,0] >= th) | (pred[:,0] <= 1-th)).sum()
    semi_ID = ind_log['Semi'][:semi_batch][:pred.shape[0]]   ## To pick small slice?
    print ('Num of included Semi:', pick_num)
    print ('Size on Semi(batch):', len(semi_ID))
    
    for i, ID in enumerate(semi_ID[:3]):
        print (i, ID,'   Prob of class 0:', pred[i,0])
    
    for i, ID in enumerate(semi_ID):    # Prob of class '0' (less than th, assign 1; more than 1-th, assign 0; else ... no label)
        if pred[i,0] >= th:       ## Likely to be class '0'
            ind_label[ID] = str(0)
            ind_log['Train'].append(ID)
            ind_log['Semi'].remove(ID)

        elif pred[i,0] <= 1-th:   ## Likely to be class '1'
            ind_label[ID] = str(1)
            ind_log['Train'].append(ID)
            ind_log['Semi'].remove(ID)

    print ('New length of Train:', len(ind_log['Train']))
    print ('New (reduced) length of Semi:', len(ind_log['Semi']))
    
    
    ## (3) New Fitting
    if retrain:
        print ('Retrain-model')
        model = call_m2()
           
    check = ModelCheckpoint('../temp/semi/model/iter' + str(semi_times + 1) + '_best.h5', 
                            monitor='val_acc', save_weights_only=True, save_best_only=True)
    train_g = DataGenerator(list_IDs = ind_log['Train'], labels = ind_label, 
                           mywv=mywv, pred=False, shuffle=True)
    val_g = DataGenerator(list_IDs = ind_log['Val'], labels = ind_label, 
                           mywv=mywv, pred=False, shuffle=False)
    print ('Start new training')
    model.fit_generator(train_g, steps_per_epoch=len(ind_log['Train']) // 200, epochs=train_ep, verbose=1, 
                        callbacks= [check], validation_data=val_g, validation_steps=len(ind_log['Val']) // 200)
    model.load_weights('../temp/semi/model/iter' + str(semi_times + 1) + '_best.h5')
    print ('Training Done')








