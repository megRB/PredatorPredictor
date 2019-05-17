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

filepath_dict = {'kaggle':   'data/detecting_insults_kaggler/train.csv','dataworld': 'data/offensive_language_dataworld/data/labeled_data_squashed.csv'}

def setup_dataframe():
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
	return df

from sklearn.linear_model import LogisticRegression

now = datetime.datetime.now()
print('[',str(now),']', 'Starting demo')

def read_accuracies():
	accuracy_store_file = open("./res/accuracy.txt", "r")
	model_accuracy = {}
	content = accuracy_store_file.readlines()
	for line in content:
		accuracy_items = line.split(':')
		model_accuracy[accuracy_items[0]] = float(accuracy_items[1][:-1])
	return model_accuracy

def normalize_dataset():
	model_accuracy = read_accuracies()
	dataset_weights = {}
	for sources in filepath_dict.keys():
		dataset_weights[sources] = []
	for sources in filepath_dict.keys():
		denom_1 = denom_2 = denom_3 = 0
		for denom_sources in filepath_dict.keys():
			denom_1 = denom_1 + model_accuracy["cnn_bow_" + denom_sources + "_accuracy"]
			denom_2 = denom_2 + model_accuracy["cnn_we_" + denom_sources + "_accuracy"]
			denom_3 = denom_3 + model_accuracy["cnn_we_pooling_" + denom_sources + "_accuracy"]
		dataset_weights[sources].append(model_accuracy["cnn_bow_" + sources + "_accuracy"]/denom_1)
		dataset_weights[sources].append(model_accuracy["cnn_we_" + sources + "_accuracy"]/denom_2)
		dataset_weights[sources].append(model_accuracy["cnn_we_pooling_" + sources + "_accuracy"]/denom_3)
	return dataset_weights
		# dataset_weights[sources].append(model_accuracy['cnn_bow_kaggle_accuracy'])

def classify_tweet(df, list_of_tweets, dataset_weights, normalization=1):
	offensive_score = [0]*len(list_of_tweets)
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
	    input_query = np.asarray(list_of_tweets)
	    input_query = vectorizer.transform(input_query)
	    
	    now = datetime.datetime.now()
	    print('[',str(now),']', 'Predicting with CNN BOW model prepared for source', source)
	    filename = './model/cnn_bow_' + source + '.h5'
	    loaded_model = load_model(filename)
	    predicted_value = loaded_model.predict(input_query)
	    cnt = 0
	    for i in predicted_value:
	    	float_equiv = i.astype(float)
	    	float_equiv = float_equiv[0]
	    	if float_equiv < 0.25:
	    		offensive_score[cnt] = offensive_score[cnt] - (1 - (1 - dataset_weights[source][0])*normalization)
	    	if float_equiv > 0.75:
	    		offensive_score[cnt] = offensive_score[cnt] + (1 - (1 - dataset_weights[source][0])*normalization)
	    	cnt = cnt+1
	    print(loaded_model.predict(input_query))
	    backend.clear_session()
	    
	    '''
	    #CNN with word embedding
	    '''

	    tokenizer = Tokenizer(num_words=5000)
	    tokenizer.fit_on_texts(sentences_train)
	    input_query = np.asarray(list_of_tweets)
	    input_query = tokenizer.texts_to_sequences(input_query)

	    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
	    maxlen = 100

	    input_query = pad_sequences(input_query, padding='post', maxlen=maxlen)
	    
	    now = datetime.datetime.now()
	    print('[',str(now),']', 'Predicting with CNN word embedded model prepared for source', source)

	    filename = './model/cnn_we_' + source + '.h5'
	    loaded_model = load_model(filename)
	    predicted_value = loaded_model.predict(input_query)
	    cnt = 0
	    for i in predicted_value:
	    	float_equiv = i.astype(float)
	    	float_equiv = float_equiv[0]
	    	if float_equiv < 0.25:
	    		offensive_score[cnt] = offensive_score[cnt] - (1 - (1 - dataset_weights[source][1])*normalization)
	    	if float_equiv > 0.75:
	    		offensive_score[cnt] = offensive_score[cnt] + (1 - (1 - dataset_weights[source][1])*normalization)
	    	cnt = cnt+1
	    print(loaded_model.predict(input_query))
	    backend.clear_session()
	    
	    #with GlobalMaxPooling1D layer to reduce number of features
	   
	    now = datetime.datetime.now()
	    print('[',str(now),']', 'Predicting with CNN word embedded model with global pooling prepared for source', source)
	    filename = './model/cnn_we_pooling_' + source + '.h5'
	    loaded_model = load_model(filename)
	    predicted_value = loaded_model.predict(input_query)
	    cnt = 0
	    for i in predicted_value:
	    	float_equiv = i.astype(float)
	    	float_equiv = float_equiv[0]
	    	if float_equiv < 0.25:
	    		offensive_score[cnt] = offensive_score[cnt] - (1 - (1 - dataset_weights[source][2])*normalization)
	    	if float_equiv > 0.75:
	    		offensive_score[cnt] = offensive_score[cnt] + (1 - (1 - dataset_weights[source][2])*normalization)
	    	cnt = cnt+1
	    print(loaded_model.predict(input_query))
	    backend.clear_session()
	for i in range(len(offensive_score)):
		if normalization == 0:
			offensive_score[i] = offensive_score[i]/6
		else:
			offensive_score[i] = offensive_score[i]/3
	print(offensive_score)

input_query_list = ['shut up bitch', 'i love my mom', 'hi honey', 'you are a hoe', 'damn mama smack that']
dataset_weights = normalize_dataset()
df = setup_dataframe()
classify_tweet(df, input_query_list, dataset_weights, normalization=1)

