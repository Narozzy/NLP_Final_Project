# -*- coding: utf-8 -*-
"""
@author Noah Rozelle and Ovidiu Mocanu
@desc  English-to-German Neural Machine Translation project for ITCS-4111,
       Natural Language Processing
"""


import string
import re
import numpy as np
from unicodedata import normalize
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector, TimeDistributed
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split


def max_line_length(lines):
    return max(len(line.split()) for line in lines)


def clean_text(lines):
    clean_lines = []
    printable = re.compile('[^{}]'.format(re.escape(string.printable)))
    
    translation_table = str.maketrans('', '', string.punctuation)
    
    for line in lines:
        cleaned = []
        for lang in line:
            lang = normalize('NFD', lang).encode('ascii', 'ignore')
            lang = lang.decode('UTF-8')
            lang = lang.split()
            lang = [word.lower() for word in lang]
            lang = [word.translate(translation_table) for word in lang]
            lang = [printable.sub('', word) for word in lang]
            lang = [word for word in lang if word.isalpha()]
            cleaned.append(' '.join(lang))
        clean_lines.append(cleaned)
    return np.array(clean_lines)

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
def create_RNN_LSTM_model(de_vocab, en_vocab, de_timesteps, en_timesteps, num_of_seqs):
    lstm_model = Sequential()
    
    lstm_model.add(Embedding(de_vocab, num_of_seqs, input_length=de_timesteps, mask_zero=True))
    lstm_model.add(LSTM(num_of_seqs))
    lstm_model.add(RepeatVector(en_timesteps))
    lstm_model.add(LSTM(num_of_seqs, return_sequences=True))
    lstm_model.add(TimeDistributed(en_vocab, activation='relu'))
    
    return lstm_model


if __name__ == '__main__':
    # Grab our doc from file, need to post where we got data from
    doc = read_document('deu.txt')
    en_de_lines = text_to_lines_array(doc)
    
    cleaned_lines = clean_text(en_de_lines)
    print(cleaned_lines[:-100])
    
    # Split our cleaned_lines into english and german arrays
    en_lines = cleaned_lines[:25000,0]
    de_lines = cleaned_lines[:25000,1]
    
    # Using Keras tokenizer method to create dictionary for our languages
    en_tokenizer = Tokenizer()
    en_tokenizer.fit_on_texts(en_lines)
    
    de_tokenizer = Tokenizer()
    de_tokenizer.fit_on_texts(de_lines)
    
    
    # Encode our lines into numerical values to train on
    
    
    