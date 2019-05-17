import pandas as pd

filepath_dict = {'yelp':   'data/sentiment_analysis/yelp_labelled.txt',
                 'amazon': 'data/sentiment_analysis/amazon_cells_labelled.txt',
                 'imdb':   'data/sentiment_analysis/imdb_labelled.txt'}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source  # Add another column filled with the source name
    df_list.append(df)

df = pd.concat(df_list)
print(df.iloc[0])

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

#splitting dataset into training and validation data

from sklearn.model_selection import train_test_split
df_yelp = df[df['source'] == 'yelp']

sentences = df_yelp['sentence'].values
y = df_yelp['label'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)
print("------------------------")
print("Training data is of size", len(sentences_train))
print("------------------------")
print("Testing data is of size", len(sentences_test))
print("------------------------")

#vectorising our data
from sklearn.feature_extraction.text import CountVectorizer

#CountVectorizer performs tokenization which separates the sentences into a set of tokens
#It additionally removes punctuation and special characters and can apply other preprocessing to each word

'''
If you want, you can use a custom tokenizer from the NLTK library with the CountVectorizer or use any number of the customizations which you can explore to improve the performance of your model.


'''
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
print(vectorizer.vocabulary_)
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)
print(X_train.toarray())


#Preparing a baseline model for comparison

#Logistic Regression

from sklearn.linear_model import LogisticRegression
print("Using logistic regression...")
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print("Accuracy:", score)


#Testing with all the datasets:

for source in df['source'].unique():
    df_source = df[df['source'] == source]
    sentences = df_source['sentence'].values
    y = df_source['label'].values

    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state=1000)

    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)
    X_train = vectorizer.transform(sentences_train)
    X_test  = vectorizer.transform(sentences_test)

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print('Accuracy for {} data: {:.4f}'.format(source, score))


#Deep Neural Networks
'''
All of those have to be then summed and passed to a function f. This function is considered the activation function and there are various 
different functions that can be used depending on the layer or the problem. It is generally common to use a rectified linear unit (ReLU) 
for hidden layers, a sigmoid function for the output layer in a binary classification problem, or a softmax function for the output layer 
of multi-class classification problems.
'''


