import numpy as np
import pandas as pd
import csv
import sys
from time import time


from keras.models import Sequential, Model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, InputLayer, Embedding
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from gensim.models import KeyedVectors




### Load and output ---- NOTE ... (1) check final rnn model  (2) Robust to shuffle

# 1. Testing Data
# 2. W2V model
# 3. preprocess list 
# 4. RNN model
# 5. Output path ... 

test_path = sys.argv[1]    #'../input/testing_data.txt'  sys.argv[1]
w2v_path = 'w2v_TO_cbow100-5-3_proc_punc_wv'          #'../temp/w2v_TO_cbow100-5-3_proc_punc_wv'
proc_list_path = 'infreq_TO_test_cb100-5-3_punc.npy'  # '../temp/infreq_TO_test_cb100-5-3_punc.npy'
rnn_path = 'iter2_best.h5'     # '../temp/RNN1/cb100-5-3_TO_pr1_punc_m2.04-0.828.h5' 'cb100-5-3_TO_pr1_punc_m2.04-0.828.h5' 'iter2_best.h5'
output_path = sys.argv[2]  # sys.argv[2]
# mode sys.argv[3]



## Input data
test_f = open(test_path)    ### 
#test_f = open('../input/test_suff.txt')
test_id, test_X = [], []
test_f.readline()    # Read the first line
for line in test_f:
    line_list = line.strip().split(',')
    test_id.append(line_list[0])
    line_list_X = ','.join(line_list[1:])
    test_X.append(line_list_X)
test_f.close()
print ('length of testing texts:',len(test_X))


## sorting (should be robust to shuffling)
test_id = np.array(test_id, dtype=int)
temp_pd = pd.DataFrame({'test_id':test_id, 'test_X':test_X})
temp_pd = temp_pd.sort_values(by=['test_id'])
test_X = list(temp_pd['test_X'])
del temp_pd
print ('First 15 as follow:','\n',test_X[:15])




## 'proc
proc1 = True
X_wseq = []
for i in range(len(test_X)):
    #wseq = text_to_word_sequence(test_X[i])
    wseq = text_to_word_sequence(test_X[i], filters='"#$%&()*+-/<=>@[\\]^_`{|}~\t\n')  # Keep ! ? . , : ;
    
    if i % 100000 == 0:
        print ('at text', i)    
    if proc1:
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




## Load W2V model and vocab_list
mywv = KeyedVectors.load(w2v_path)  ###
vocab_list = mywv.index2word
print ('vocab size', len(vocab_list))



## Setting
size = len(test_X)
max_length = 40 
emb_dim = 100
print ('Data size, length of a text, embedded dim:', size, max_length, emb_dim)




## PreProcess
proc_infreq_list = np.load(proc_list_path)   ###
print (len(proc_infreq_list))

t0 = time()
print ('Num of text needed to be preprocess(Infrequency words):', len(proc_infreq_list))
for text_i in proc_infreq_list:  # Smaller size for trial

    ## Filter out infreq words
    X_wseq[text_i] = [ele for ele in X_wseq[text_i] if ele in vocab_list]
    #X_wseq[text_i] = [ele if ele in vocab_list else 'InFreq_Token' for ele in X_wseq[text_i]]
    
print ('Time Consumption:', time()-t0)


X_testing = np.zeros([size, max_length, emb_dim])
t0 = time()
for text_i in range(size):    
    ## Out of 40 and 0 (In some case...filter out infreq words make text empty)
    if len(X_wseq[text_i]) == 0:
        pad_emb_text = np.zeros([max_length, emb_dim])
        X_testing[text_i,:,:] = pad_emb_text
    else:    
        emb_text = mywv[X_wseq[text_i]]
        num_pad = 40 - len(X_wseq[text_i])
        pad_emb_text = np.pad(emb_text,pad_width=((num_pad,0),(0,0)), mode='constant')
        X_testing[text_i,:,:] = pad_emb_text
        
print ('Time Consumption:', time()-t0)





## m2 

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
model.load_weights(rnn_path)   ###
##




## Predicting and Output
t0 = time()
pred = model.predict_classes(X_testing, verbose=1, batch_size=200)
print ('First 10 prediction:',pred[:10])
print ('Prediction length:', (pred==1).sum())
print ('time consumption:', time()-t0)


## Output
t0 = time()
filename = output_path    ###
f1 = open(filename, 'w+')
f1_text = csv.writer(f1,delimiter=',',lineterminator='\n')
f1_text.writerow(['id','label'])
for i in range(size):
    f1_text.writerow([int(i), pred[i]]) # pred[i]
f1.close()
print ('Time consumption:', time()-t0)





