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
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
import numpy as np
import pickle
import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import datetime
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm


#usage: tfidf(df_source, sentences_train, sentences_test) 
def tfidf(trainDF, train_x, valid_x):
    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(trainDF['tweet'])
    xtrain_tfidf =  tfidf_vect.transform(train_x)
    xvalid_tfidf =  tfidf_vect.transform(valid_x)

    # ngram level tf-idf 
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram.fit(trainDF['tweet'])
    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
    xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

    # characters level tf-idf
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram_chars.fit(trainDF['tweet'])
    xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
    xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 

def logistic_regression_modeller(X_train, y_train, X_test, y_test, source):
    now = datetime.datetime.now()
    print('[',str(now),']', 'Logistic Regression started for source', source)
    print("\nUsing Logistic Regression")
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print("----------------------------------------------------------------")
    print('Accuracy for {} data: {:.4f} using Logistic Regression'.format(source, score))
    print("----------------------------------------------------------------")
    now = datetime.datetime.now()
    print('[',str(now),']', 'Regression training completed for source', source)
    return classifier

def naive_bayes_modeller(X_train, y_train, X_test, y_test, source):
    now = datetime.datetime.now()
    print('[',str(now),']', 'Naive Bayes started for source', source)
    print("\nUsing Naive Bayes")
    score, classifier = train_model(naive_bayes.MultinomialNB(), X_train, y_train, X_test, y_test)
    print("----------------------------------------------------------------")
    print('Accuracy for {} data: {:.4f} using Naive Bayes'.format(source, score))
    print("----------------------------------------------------------------")
    now = datetime.datetime.now()
    print('[',str(now),']', 'Naive Bayes training completed for source', source)
    return classifier

def train_model(classifier, feature_vector_train, label, feature_vector_test, y_test, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_test)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    return metrics.accuracy_score(predictions, y_test), classifier

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


filepath_dict = {'kaggle':   'data/detecting_insults_kaggler/train.csv','dataworld': 'data/offensive_language_dataworld/data/labeled_data_squashed.csv'}

df_list = []

source = "kaggle"
filepath = filepath_dict["kaggle"]
#df = pd.read_csv(filepath, names=['rev_id', 'comment year','logged_in',   'ns',  'sample',  'split'], sep='\t')
df = pd.read_csv(filepath, names=['label', 'date','tweet'], sep=',',header=0)
df['source'] = source  # Add another column filled with the source name
df_list.append(df)
df = pd.concat(df_list)
df = df.drop(['date'], axis=1)

source = "dataworld"
filepath = filepath_dict["dataworld"]
#df = pd.read_csv(filepath, names=['rev_id', 'comment year','logged_in',   'ns',  'sample',  'split'], sep='\t')
df = pd.read_csv(filepath, names=['id', 'count','hate_speech', 'offensive_language','neither','class', 'tweet', 'label'], sep=',',header=0)
df['source'] = source  # Add another column filled with the source name
df_list.append(df)
df = pd.concat(df_list)

df = df.drop(['count','hate_speech', 'offensive_language','neither'], axis=1)

from sklearn.linear_model import LogisticRegression

now = datetime.datetime.now()
print('[',str(now),']', 'Starting demo')

for source in df['source'].unique():
    if source == "kaggle":
        df_source = df[df['source'] == source]
        sentences = df_source['tweet'].values
        y = df_source['label'].values
    elif source == "dataworld":
        df_source = df[df['source'] == source]
        sentences = df_source['tweet'].values
        y = df_source['class'].values
        # print(y)
    now = datetime.datetime.now()
    print('[',str(now),']', 'Processing started for source', source)
    print("----------------------------------------------------------------")
    print("----------------------------------------------------------------")
    print(source.upper(), "SOURCE")
    print("----------------------------------------------------------------")
    print("----------------------------------------------------------------")
    #splitting dataset into training and validation data
    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state=1000)
    
    '''
    CNN with BOW
    '''

    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)
    X_train = vectorizer.transform(sentences_train)
    X_test  = vectorizer.transform(sentences_test)
    now = datetime.datetime.now()
    print('[',str(now),']', 'Evaluating CNN BOW model prepared for source', source)
    filename = './model/cnn_bow_' + source + '.h5'
    loaded_model = load_model(filename)
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    loss, accuracy = loaded_model.evaluate(X_test, y_test, verbose=False)
    print("Accuracy: %.2f%%" % (accuracy*100))
    now = datetime.datetime.now()
    print('[',str(now),']', 'CNN evaluation completed for source', source)
    backend.clear_session()
    # accuracy_store_file.write('cnn_bow_' + source + '_accuracy:' + str(accuracy) + '\n')
    '''
    #CNN with word embedding
    '''
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sentences_train)
    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)

    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
    maxlen = 100

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    
    now = datetime.datetime.now()
    print('[',str(now),']', 'Evaluating CNN word embedded model prepared for source', source)

    filename = './model/cnn_we_' + source + '.h5'
    loaded_model = load_model(filename)
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = loaded_model.evaluate(X_test, y_test, verbose=False)
    print("Accuracy: %.2f%%" % (score[1]*100))
    now = datetime.datetime.now()
    print('[',str(now),']', 'CNN evaluation completed for source', source)
    backend.clear_session()
    # accuracy_store_file.write('cnn_we_' + source + '_accuracy:' + str(accuracy) + '\n')
    
    #with GlobalMaxPooling1D layer to reduce number of features
   
    now = datetime.datetime.now()
    print('[',str(now),']', 'Evaluating CNN word embedded model with global pooling prepared for source', source)
    filename = './model/cnn_we_pooling_' + source + '.h5'
    loaded_model = load_model(filename)
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    loss, accuracy = loaded_model.evaluate(X_test, y_test, verbose=False)
    print("Accuracy: %.2f%%" % (accuracy*100))
    now = datetime.datetime.now()
    print('[',str(now),']', 'CNN evaluation completed for source', source)
    backend.clear_session()
    # accuracy_store_file.write('cnn_we_pooling_' + source + '_accuracy:' + str(accuracy) + '\n')
