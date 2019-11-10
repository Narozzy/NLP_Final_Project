# -*- coding: utf-8 -*-
"""
@author Noah Rozelle and Ovidiu Mocanu
@desc  English-to-German Neural Machine Translation project for ITCS-4111,
       Natural Language Processing
"""


import string
import re
import numpy as np
# Part of python standard library, provides a way to convert accented characters
# with their corresponding unicode interpretation, this is to make sure our
# model scales well
from unicodedata import normalize
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector, TimeDistributed, GRU
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def clean_text(lines):
    clean_lines = []
    
    # Regular expression for printable characters
    printable = re.compile('[^{}]'.format(re.escape(string.printable)))
    
    # remove punctuation from our texts so we can focus on translating words
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
# @desc Encode our lines with the tokenizers corresponding value, then we pad w/ zeros
# =============================================================================
def encode_lines(txt, size, tkn):
    print(size)
    sequences = tkn.texts_to_sequences(txt)
    # add zero to sequences so that they become the same size, necessary for training
    for seq in sequences:
        while len(seq) < size:
            seq.append(0)
    return sequences


# =============================================================================
# @author Noah Rozelle - 801028077
# @desc one hot encoding for our train_y and test_y as it is required for keras RNN models
# =============================================================================
def one_hot_encoding(seqs, vocab):
    ys = []

    if type(seqs) != 'numpy.ndarray':
        seqs = np.array(seqs)
    
    for seq in seqs:
        one_hot_encode = to_categorical(seq, num_classes=vocab)
        ys.append(one_hot_encode)
    ys = np.array(ys)
    ys = ys.reshape(seqs.shape[0], seqs.shape[1], vocab)
    return ys
        

# =============================================================================
# @author Noah Rozelle - 801028077
# @desc   Creation of model number 1, simple RNN for purpose of translation (GRU)
# @params:
    #n = Number of Hidden Nodes
    #en_len = Length of english sentence
    #de_len = Length of german sentence
# =============================================================================
def create_GRU_RNN_model(de_vocab, en_vocab, de_timesteps, en_timesteps, num_of_seqs):
    gru_model = Sequential()
    gru_model.add(Embedding(de_vocab, num_of_seqs, input_length=de_timesteps,mask_zero=True))
    gru_model.add(GRU(num_of_seqs))
    gru_model.add(RepeatVector(en_timesteps))
    gru_model.add(GRU(num_of_seqs, return_sequences=True))
    gru_model.add(TimeDistributed(Dense(en_vocab, activation='sigmoid')))
    return gru_model

# =============================================================================
# @author Noah Rozelle - 801028077
# @desc Creation of model number 2, RNN w/ LSTM for purpose of translation
# =============================================================================
def create_RNN_LSTM_model(de_vocab, en_vocab, de_timesteps, en_timesteps, num_of_seqs):
    # Core layer from Keras for RNN implementations
    lstm_model = Sequential()
    
    # the layers making up our RNN LSTM model
    lstm_model.add(Embedding(de_vocab, num_of_seqs, input_length=de_timesteps,mask_zero=True))
    lstm_model.add(LSTM(num_of_seqs))
    lstm_model.add(RepeatVector(en_timesteps))
    lstm_model.add(LSTM(num_of_seqs, return_sequences=True))
    lstm_model.add(TimeDistributed(Dense(en_vocab, activation='relu')))
    
    return lstm_model


if __name__ == '__main__':
    # Grab our doc from file, need to post where we got data from
    doc = read_document('deu.txt')
    en_de_lines = text_to_lines_array(doc)
    
    cleaned_lines = clean_text(en_de_lines)
    
    # Split our cleaned_lines into english and german arrays
    en_lines = cleaned_lines[:25000,0]
    de_lines = cleaned_lines[:25000,1]
    
    # Using Keras tokenizer method to create dictionary for our languages
    en_tokenizer = Tokenizer()
    en_tokenizer.fit_on_texts(en_lines)
    
    de_tokenizer = Tokenizer()
    de_tokenizer.fit_on_texts(de_lines)
    
    # Define our vocab size
    en_size = len(en_tokenizer.word_index) + 1
    de_size = len(de_tokenizer.word_index) + 1
    
    # Encode our lines into numerical values to train on
    # Our 'Y', we pass in the cleaned lines for english samples, the max length of english sentences, as well as our tokenizer 
    en_seq = encode_lines(en_lines, max([len(w.split()) for w in en_lines]), en_tokenizer)
    # Our 'X', same as above, but for our german phrases
    de_seq = encode_lines(de_lines, max([len(w.split()) for w in de_lines]), de_tokenizer)
    
    # now we must split our set into training and testing
    train_x, test_x, train_y, test_y = train_test_split(de_seq, en_seq, test_size=0.25, random_state=42, shuffle=False)
    
    # encode our ys
    train_y = one_hot_encoding(train_y, en_size)
    test_y = one_hot_encoding(test_y, en_size)
    
    # change each into numpy arrays
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    
    # Now we can create our LSTM Model
#    lstm = create_RNN_LSTM_model(de_size, en_size, max([len(w.split()) for w in de_lines]), max([len(w.split()) for w in en_lines]), 256)
#    lstm.compile(loss='mean_squared_error', optimizer='sgd')
#    print(lstm.summary())
#    lstm.fit(train_x, train_y, epochs=30, batch_size=64, validation_data=(test_x, test_y))
    
    # Now we can create our GRU model
    gru = create_GRU_RNN_model(de_size, en_size, max([len(w.split()) for w in de_lines]), max([len(w.split()) for w in en_lines]), 256)
    gru.compile(loss='mean_squared_error', optimizer='sgd')
    print(gru.summary())
    gru.fit(train_x, train_y, epochs=30, batch_size=64, validation_data=(test_x, test_y))