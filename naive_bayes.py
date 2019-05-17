#!/usr/bin/env python -W ignore::DeprecationWarning
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras import layers
from keras import backend
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import datetime
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
def train_model(classifier, feature_vector_train, label, feature_vector_test, y_test, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    return metrics.accuracy_score(predictions, valid_y)
filepath_dict = {'kaggler':   'data/detecting_insults_kaggler/train.csv','dataworld': 'data/offensive_language_dataworld/data/labeled_data.csv'}

df_list = []

source = "kaggler"
filepath = filepath_dict["kaggler"]
#df = pd.read_csv(filepath, names=['rev_id', 'comment year','logged_in',   'ns',  'sample',  'split'], sep='\t')
df = pd.read_csv(filepath, names=['label', 'date','tweet'], sep=',',header=0)
df['source'] = source  # Add another column filled with the source name
df_list.append(df)
df = pd.concat(df_list)
df = df.drop(['date'], axis=1)

source = "dataworld"
filepath = filepath_dict["dataworld"]
#df = pd.read_csv(filepath, names=['rev_id', 'comment year','logged_in',   'ns',  'sample',  'split'], sep='\t')
df = pd.read_csv(filepath, names=['id', 'count','hate_speech', 'offensive_language','neither','class', 'tweet'], sep=',',header=0)
df['source'] = source  # Add another column filled with the source name
df_list.append(df)
df = pd.concat(df_list)

df = df.drop(['count','hate_speech', 'offensive_language','neither'], axis=1)

for source in df['source'].unique():
    if source == "kaggler":
        df_source = df[df['source'] == source]
        sentences = df_source['tweet'].values
        y = df_source['label'].values
    elif source == "dataworld":
        df_source = df[df['source'] == source]
        sentences = df_source['tweet'].values
        y = df_source['class'].values
    now = datetime.datetime.now()
    print('[',str(now),']', 'Processing started for source', source)
    print("----------------------------------------------------------------")
    print("----------------------------------------------------------------")
    print(source.upper(), "SOURCE")
    #splitting dataset into training and validation data
    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state=1000)
    print("----------------------------------------------------------------")
    print("Training data is of size", len(sentences_train))
    print("Testing data is of size", len(sentences_test))
    print("----------------------------------------------------------------")

    #vectorising our data


    #CountVectorizer performs tokenization which separates the sentences into a set of tokens
    #It additionally removes punctuation and special characters and can apply other preprocessing to each word


    now = datetime.datetime.now()
    print('[',str(now),']', 'Vectorization started for source', source)
    
    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)
    X_train = vectorizer.transform(sentences_train)
    X_test  = vectorizer.transform(sentences_test)

    accuracy = train_model(naive_bayes.MultinomialNB(), X_train, y_train, X_test, y_test)
    print("NB accuracy: ", accuracy)