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


from keras.models import Sequential
from keras import layers

input_dim = X_train.shape[1]  # Number of features

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()