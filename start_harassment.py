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
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import os
import numpy as np
import pickle
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
    return classifier, score

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
    return classifier, score

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

'''
print(df.iloc[1])
for i in range(len(df)):
    print(df.iloc[i])
'''
'''
#to vectorise words into feature vector

from sklearn.feature_extraction.text import CountVectorizer
sentences = ['John likes ice cream', 'John hates chocolate.']
vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(sentences)
vectorizer.vocabulary_
print(vectorizer.vocabulary_)

#Bag of Words model
vectorizer.transform(sentences).toarray()
print(vectorizer.transform(sentences).toarray())
'''

#splitting dataset into training and validation data

#vectorising our data


#CountVectorizer performs tokenization which separates the sentences into a set of tokens
#It additionally removes punctuation and special characters and can apply other preprocessing to each word

'''
#If you want, you can use a custom tokenizer from the NLTK library with the CountVectorizer or use any number of the customizations which you can explore to improve the performance of your model.

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
# print(vectorizer.vocabulary_)
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)
# print(X_train.toarray())

#Preparing a baseline model for comparison

#Logistic Regression

from sklearn.linear_model import LogisticRegression
print("Using logistic regression...")
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print("Accuracy:", score)
'''

#Logistic Regression baseline model with all the datasets:


from sklearn.linear_model import LogisticRegression

optimal_epoch = {"kaggle": 4, "dataworld": 5}
optimal_epoch_max_pool = {"kaggle": 3, "dataworld": 8}

now = datetime.datetime.now()
print('[',str(now),']', 'Starting demo')


def train_models():
    accuracy_store_file = open("./res/accuracy.txt", "w+")
    for source in df['source'].unique():
        if source == "kaggle":
            continue
            df_source = df[df['source'] == source]
            sentences = df_source['tweet'].values
            y = df_source['label'].values
        elif source == "dataworld":
            df_source = df[df['source'] == source]
            sentences = df_source['tweet'].values
            # print(sentences)
            y = df_source['label'].values
            # print(y)
         
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
    
        
        #Preparing a baseline model for comparison
        #Logistic Regression
        
        lr_classifier, accuracy = logistic_regression_modeller(X_train, y_train, X_test, y_test, source)
        accuracy_store_file.write('lr_' + source + '_accuracy:' + str(accuracy) + '\n')
        print("Accuracy metrics saved to file")
        
        #Preparing a baseline model for comparison
        #Naive Bayes

        nb_classifier, accuracy = naive_bayes_modeller(X_train, y_train, X_test, y_test, source)
        accuracy_store_file.write('nb_' + source + '_accuracy:' + str(accuracy) + '\n')
        print("Accuracy metrics saved to file")

        
        
        now = datetime.datetime.now()
        print('[',str(now),']', 'CNN training started for source', source)
        print("\nUsing Deep Neural Networks without word embedding")
        
        input_dim = X_train.shape[1]  # Number of features

        model = Sequential()
        model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        print("Fitting model now. It may take a while...")
        history = model.fit(X_train, y_train,epochs=optimal_epoch[source],verbose=False,validation_data=(X_test, y_test),batch_size=10)
        loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
        print("----------------------------------------------------------------")
        print("Training Accuracy for {} data: {:.4f} using CNN".format(source, accuracy))
        loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
        print("Testing Accuracy for {} data: {:.4f} using CNN".format(source, accuracy))
        yhat_classes = model.predict_classes(X_test, verbose=0)
        yhat_classes = yhat_classes[:, 0]
        precision = precision_score(y_test, yhat_classes)
        recall = recall_score(y_test, yhat_classes)
        print("Testing Precision for {} data: {:.4f} using CNN".format(source, precision))
        print("Testing Recall for {} data: {:.4f} using CNN".format(source, recall))
        print("----------------------------------------------------------------")
        print("----------------------------------------------------------------")
        now = datetime.datetime.now()
        print('[',str(now),']', 'CNN training completed for source', source)
        filename = './model/cnn_bow_' + source + '.h5'
        model.save(filename)
        print("Saved model to disk")
        accuracy_store_file.write('cnn_bow_' + source + '_accuracy:' + str(accuracy) + '\n')
        accuracy_store_file.write('cnn_bow_' + source + '_precision:' + str(precision) + '\n')
        accuracy_store_file.write('cnn_bow_' + source + '_recall:' + str(recall) + '\n')
        print("Accuracy metrics saved to file")
        # y = model.predict(input_query)
        # print("Output")
        # print(y)
        
        #A good way to see when the model starts overfitting is when the loss of the validation data starts rising again. This tends to be a good point to stop the model. 
        
        plot_history(history)
        backend.clear_session()
        #CNN with word embedding
        
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(sentences_train)

        X_train = tokenizer.texts_to_sequences(sentences_train)
        X_test = tokenizer.texts_to_sequences(sentences_test)
        vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
        print("Word embedding example weightage")
        print(sentences_train[4])
        print(X_train[4])
        print('\n')
        print(sentences_train[53])
        print(X_train[53])
        print('\n')
        print(sentences_train[10])
        print(X_train[10])
        
        #for word in ['the', 'all', 'happy', 'sad']: print('{}: {}'.format(word, tokenizer.word_index[word]))

        maxlen = 100

        X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
        
        
        now = datetime.datetime.now()
        print('[',str(now),']', 'CNN training with word embedding started for source', source)
        print("\nUsing Deep Neural Networks with word embedding")
        
        embedding_dim = 50
        model = Sequential()
        model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
        model.add(layers.Flatten())
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        print("Fitting model now. It may take a while...")
        history = model.fit(X_train, y_train, epochs=5, verbose=False, validation_data=(X_test, y_test), batch_size=10)
        loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
        print("----------------------------------------------------------------")
        print("Training Accuracy for {} data: {:.4f} using CNN".format(source, accuracy))
        loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
        print("Testing Accuracy for {} data: {:.4f} using CNN".format(source, accuracy))
        yhat_classes = model.predict_classes(X_test, verbose=0)
        yhat_classes = yhat_classes[:, 0]
        precision = precision_score(y_test, yhat_classes)
        recall = recall_score(y_test, yhat_classes)
        print("Testing Precision for {} data: {:.4f} using CNN".format(source, precision))
        print("Testing Recall for {} data: {:.4f} using CNN".format(source, recall))
        print("----------------------------------------------------------------")
        print("----------------------------------------------------------------")
        now = datetime.datetime.now()
        print('[',str(now),']', 'CNN training completed for source', source)
        filename = './model/cnn_we_' + source + '.h5'
        model.save(filename)
        print("Saved model to disk")
        accuracy_store_file.write('cnn_we_' + source + '_accuracy:' + str(accuracy) + '\n')
        accuracy_store_file.write('cnn_we_' + source + '_precision:' + str(precision) + '\n')
        accuracy_store_file.write('cnn_we_' + source + '_recall:' + str(recall) + '\n')
        print("Accuracy metrics saved to file")
        plot_history(history)
        # predicted = model.predict(X_test)
        # predicted = np.argmax(predicted, axis=1)
        # print(accuracy_score(y_test, predicted))
        backend.clear_session()
        
        #with GlobalMaxPooling1D layer to reduce number of features
        '''
        embedding_dim = 50

        model = Sequential()
        model.add(layers.Embedding(input_dim=vocab_size, 
                                   output_dim=embedding_dim, 
                                   input_length=maxlen))
        model.add(layers.GlobalMaxPool1D())
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        print("Fitting model now. It may take a while...")
        history = model.fit(X_train, y_train,
                        epochs=optimal_epoch_max_pool[source],
                        verbose=False,
                        validation_data=(X_test, y_test),
                        batch_size=10)
        loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))
        now = datetime.datetime.now()
        print('[',str(now),']', 'Training completed for source', source)
        filename = './model/cnn_we_pooling_' + source + '.h5'
        model.save(filename)
        print("Saved model to disk")
        accuracy_store_file.write('cnn_we_pooling_' + source + '_accuracy:' + str(accuracy) + '\n')
        print("Accuracy metrics saved to file")
        plot_history(history)
        backend.clear_session()
        '''
train_models()
