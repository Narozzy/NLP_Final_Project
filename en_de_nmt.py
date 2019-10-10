# -*- coding: utf-8 -*-
"""
@author Noah Rozelle and Ovidiu Mocanu
@desc  English-to-German Neural Machine Translation project for ITCS-4111,
       Natural Language Processing
"""

import string
import re
import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector, Dropout
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

# =============================================================================
# @author Noah Rozelle - 801028077
# @desc   read document given filename
# @params
    #filename = Name of our dataset file
# =============================================================================
def read_document(filename):
    with open(filename, mode='r', encoding='utf-8') as f:
        text = f.read()
    return text

# =============================================================================
# @author Noah Rozelle - 801028077
# @desc   Split a document into lines and return as np.array
# @params
    #txt=document
# =============================================================================
def text_to_lines_array(txt):
    sentences = txt.strip().split('\n')
    sentences = [i.split('\t') for i in sentences]
    return np.array(sentences)

# =============================================================================
# @author Noah Rozelle - 801028077
# @desc   Tokenizer an array of strings
# @params
    #arr = array of strings
# =============================================================================
def tokenize_strings(tkn, arr):
    tkn.fit_on_texts(arr)
    new_arr = tkn.texts_to_sequences(arr)
    return np.array(new_arr)

# =============================================================================
# @author Noah Rozelle - 801028077
# @desc   Create one-hot-array reps of sentences
# @params
    #arr = language array
    #  n = max size of embedding
# =============================================================================
def create_one_hot_array(arr, n):
    one_hot_embed_line = np.zeros(shape=(1,n))
    for i in arr:
        one_hot_embed_line[0,i-1] = 1
    return one_hot_embed_line
# =============================================================================
# @author Noah Rozelle - 801028077
# @desc   Creation of model number 1, simple RNN for purpose of translation (GRU)
# @params:
    #n = Number of Hidden Nodes
    #en_len = Length of english sentence
    #de_len = Length of german sentence
# =============================================================================
def create_RNN_model(n, en_len, de_len):
    rnn_model = Sequential()
    pass

# =============================================================================
# @author Noah Rozelle - 801028077
# @desc Creation of model number 2, RNN w/ LSTM for purpose of translation
# =============================================================================
def create_RNN_LSTM_model(n, train_shape, test_shape):
    lstm_model = Sequential()
    
    lstm_model.add(LSTM(128, input_shape=train_shape, activation='relu', return_sequences=True))
    lstm_model.add(Dropout(0.2))
    
    lstm_model.add(LSTM(128, activation='relu'))
    lstm_model.add(Dropout(0.2))
    
    lstm_model.add(Dense(32, activation='relu'))
    lstm_model.add(Dropout(0.2))
    
    lstm_model.add(Dense(10, activation='relu'))
    
    optimizer = Adam(decay=1e-5)
    lstm_model.compile(loss='sparse_categorical_crossentropy',
                       optimizer=optimizer,
                       metrics=['accuracy'])
    return lstm_model

# NOTE: Will need to reduce the number of lines in the document
if __name__ == '__main__':
    # Grab our doc from file, need to post where we got data from
    doc = read_document('deu.txt')
    en_de_lines = text_to_lines_array(doc)
    
    # Split our en_de_lines into english and german arrays
    en_lines = en_de_lines[:10000,0]
    de_lines = en_de_lines[:10000,1]
    
    
    print(en_lines.shape)
    print(de_lines.shape)
    # Credit: "Recurrent Neural Networds by Example in Python" by Will Koehrsen
    # URL: https://towardsdatascience.com/recurrent-neural-networks-by-example-in-python-ffd204f99470
    tkn = Tokenizer(num_words=None,
                    filters='#$%&()*+-<=>@[\\]^_`{|}~\t\n',
                    lower=False, split=' ')
    
    # Convert the lines of text into integer sequences
    en_tokenized = tokenize_strings(tkn, en_lines)
    de_tokenized = tokenize_strings(tkn, de_lines)
    
    print(en_tokenized.shape)
    print(de_tokenized.shape)
    # Grab our token.index_word
    token_to_word = tkn.index_word
    
    # Create one hot embedded arrays for lines
    for i,(en_line, de_line) in enumerate(zip(en_tokenized, de_tokenized)):
        en_tokenized[i] = create_one_hot_array(en_line, max([i for i in token_to_word]))
        de_tokenized[i] = create_one_hot_array(de_line, max([i for i in token_to_word]))
    
    # Split our training set into training set and testing set, random_state 42 because that is the answer to life
    # x_train/x_test are english while y_train/y_test are german
    x_train, y_train, x_test, y_test = train_test_split(en_tokenized, de_tokenized, test_size=0.25, random_state=42, shuffle=False)
    
    print(type(x_train))
    
    rnn_lstm = create_RNN_LSTM_model(max([i for i in token_to_word]),
                                     train_shape=x_train.shape,
                                     test_shape=y_train.shape)
    
    rnn_lstm.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
    