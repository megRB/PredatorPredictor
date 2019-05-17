import pandas as pd
from bs4 import BeautifulSoup
import re
import html
import dataDecoder
import itertools


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

def load_slang():
	slang_store_file = open("./res/slang.txt", "r")
	slang_dictionary = {}
	content = slang_store_file.readlines()
	for line in content:
		slang_item = line.split('=')
		slang_dictionary[slang_item[0]] = str(slang_item[1][:-1])
	return (slang_dictionary)

def cleanup(text):
	apostrophe = {"re": "are", "nt": "not", "s": "is", 'd': 'would', 'll': 'will', 've': 'have'}
	processed_file = open("./res/processes_tweets.txt", "w+")
	print("Original Tweet")
	print(text)
	print("\n")
	# example1 = BeautifulSoup(df.tweet[279], 'lxml')
	# print(example1.get_text())	

	#Cleaning HTML
	#print("Cleaning html...")
	text = html.unescape(text)
	
	#Decoding unicode symbols
	text = dataDecoder.unicodetoascii(text)

	#Cleaning tags and mentions
	#print("Cleaning tags and mentions...")
	text = re.sub(r'@[A-Za-z0-9_]+','',text) 
	
	#Cleaning URLS
	#print("Cleaning urls...")
	text = re.sub('https?://[A-Za-z0-9./]+','',text)
		
	#Cleaning hashtags and Symbols
	#print("Cleaning hashtags and symbols...")
	text = re.sub("[:\"#]+", " ", text)
	
	#Cleaning extra spaces
	#print("Cleaning extra space...")
	text = re.sub("[ ]+", " ", text)
	
	#Standardize words
	text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text)) 

	#Remove excess newline
	text = re.sub("\n+", ". ", text)


	#Cleaning apostrophe short forms
	#print("Cleaning apostrophe words...")
	words = re.findall(r"\w+[\']\w+", text)
	for word in words:
		print(word)
		split_words = word.split("\'")
		print(split_words)
		for i in range(len(split_words)):
			if split_words[i] in apostrophe.keys():
				split_words[i] = apostrophe[split_words[i]]
			else:
				pass
		new_word = " ".join(split_words)
		text = text.replace(word, new_word)
			# print(text)
			# print("\n")

			#Cleaning slang
			#print("Cleaning slang...")
	slang_dictionary=load_slang()
	for word in re.findall('\w+',text):
		if word.upper() in slang_dictionary.keys():
			new_text = slang_dictionary[word.upper()]
			print(word, new_text)
			text = re.sub(word, new_text, text)
	return text

# df = setup_dataframe()
# cleanup(df)
# load_slang()