import os
import re
import gdown
import numpy
import string
import numpy as np
import sys
import pandas as pd
import tensorflow as tf
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
from absl import logging
from tensorflow import keras
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation
from keras.callbacks import LambdaCallback
import tensorflow.keras.backend as K
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding


"""## **Data preparation - _Generating Corpus_**"""

# Download data from Google drive

'''
ORIGINAL DATASET URL:
    https://raw.githubusercontent.com/maxim5/stanford-tensorflow-tutorials/master/data/arxiv_abstracts.txt
'''

url = ' https://drive.google.com/uc?id=1YTBR7FiXssaKXHhOZbUbwoWw6jzQxxKW'
output = 'corpus.txt'
gdown.download(url, output, quiet=False)

# sentence_length = 40

# Read local file from directory
with open('corpus.txt') as subject:
    cache = subject.readlines()
translator = str.maketrans('', '', string.punctuation)  # Remove punctuation
lines = [doc.lower().translate(translator) for doc in cache]  # Switch to lower case

# PREVIEW OUTPUT ::

# print(lines[0][:101])

# Generate a list of single/independent words

vocabulary = list(set(' '.join(lines).replace('\n', '').split(' ')))
primary_store = {}
for strings, texts in enumerate(vocabulary):
    primary_store[texts] = strings

# PREVIEW OUTPUT ::

# print(vocabulary[:50])


# Splitting data into Train sets and test sets

X = []
y = []

for c in lines:
    xxxx = c.replace('\n', '').split(' ')
    X.append(' '.join(xxxx[:-1]))  # X from the corpus

    yyyy = [0 for i in range(len(vocabulary))]  # Generate Y from the Vocabulary
    yyyy[primary_store[xxxx[-1]]] = 1
    y.append(yyyy)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
y_test = numpy.array(y_test)
y_train = numpy.array(y_train)

# PREVIEW OUTPUT ::

# print(X_train[:10])
# print(y_train[:10])
# print(X_test[:10])
# print(y_test[:10])

"""## **Embeddings!**"""

# Import the Universal Sentence Encoder's TF Hub module (Here we're making use of version 4)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
appreciate = hub.load(module_url)


# Making it easier - Function for embedding
def embed(goodness):
    return appreciate(goodness)


# REVIEW OUTPUT ::

# appreciate.variables

# Wrapping up with the U-S-E

X_train = embed(X_train)
X_test = embed(X_test)
X_train = X_train.numpy()
X_test = X_test.numpy()

# PREVIEW OUTPUT ::

# print(X_train[:10])
# print(y_train[:10])
# print(X_test[:10])
# print(y_test[:10])
# print(X_train.shape, X_test.shape, y_test.shape, y_train.shape)

"""# **Building the model**"""

model = Sequential()
# model.add(Embedding(input_dim=len(vocabulary), output_dim=100))
model = Sequential()
# model.add(LSTM(units=100, input_shape=[512]))
model.add(Dense(512, input_shape=[512], activation='relu'))
model.add(Dense(units=len(vocabulary), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()

# Training the model.

model.fit(X_train, y_train, batch_size=512, shuffle=True, epochs=20, validation_data=(X_test, y_test),
          callbacks=[LambdaCallback()])

"""#**Unto the tests!**"""


# Create function to predict and show detailed output

def next_word(collection=[], extent=1):
    for item in collection:
        text = item
        for i in range(extent):
            prediction = model.predict(x=embed([item]).numpy())
            idx = np.argmax(prediction[-1])
            item += ' ' + vocabulary[idx]

            print(text + ' --> ' + item + '\nNEXT WORD: ' + item.split(' ')[-1] + '\n')


# Tests - please feel free to explore

single_text = ['and some other essential']

next_word(single_text)

# Testing on a collection of words

text_collection = ['deep convolutional', 'simple and effective', 'a nonconvex', 'a']

next_word(text_collection)

text_collection1 = ['game-theoretic', 'Methods from convex optimization are widely']

next_word(text_collection1)

# Storing data

vocabulary = numpy.array(vocabulary)
numpy.save('./vocabulary.npy', vocabulary)
model.save('./NWP-USE')
