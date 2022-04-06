# Part 3: Mining text data.

import pandas as pd
import numpy as np
from collections import Counter
import requests
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import time


# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_3(data_file):
	return pd.read_csv(data_file, encoding='latin-1')


# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
	return(list(df['Sentiment'].unique()))

# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
	return(df['Sentiment'].value_counts()[1])

# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
	df1 = df.loc[df['Sentiment'] == 'Extremely Positive']
	df1 = df1.groupby(by="TweetAt").count()
	return(df1.idxmax()[1])

# Modify the dataframe df by converting all tweets to lower case. 
def lower_case(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.lower()
	return(df)

# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
	df.OriginalTweet = df.OriginalTweet.str.replace('[^a-zA-Z ]', ' ', regex=True)
	return(df)

# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
	df.OriginalTweet = df.OriginalTweet.str.replace(' +', ' ', regex=True).str.strip()
	return(df)

# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
	df.OriginalTweet = df.OriginalTweet.str.split()
	return(df)

# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
	count = 0
	for i in range (len(tdf)):
		count += len(tdf.OriginalTweet[i])
	return(count)

# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
	return(len(set(np.concatenate(tdf.OriginalTweet))))

# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf,k):
	counter=Counter(tdf.OriginalTweet[0])
	for i in range (1,len(tdf)):
		counter+=Counter(tdf.OriginalTweet[i])
	return(counter.most_common(k))

# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt

def remove_stop_words(tdf):
	url = 'https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt'
	r = requests.get(url, allow_redirects=True)
	stopwords = set(str(r.content).replace("\\'","'").split("\\n"))

	tdf.OriginalTweet = tdf.OriginalTweet.apply(lambda row: [word for word in row if not word in stopwords and len(word) > 2])
	return(tdf)

# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
	porter_stemmer = PorterStemmer()
	tdf.OriginalTweet = tdf.OriginalTweet.apply(lambda i: [porter_stemmer.stem(word) for word in i])
	return(tdf)


# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):

	x_train = df.OriginalTweet.to_list()
	y_train = df.Sentiment.to_list()

	
	url = 'https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt'
	r = requests.get(url, allow_redirects=True)
	stopwords = set(str(r.content).replace("\\'","'").split("\\n"))
	
	vec = CountVectorizer(stop_words=stopwords, ngram_range=(2,4))
	x_train = vec.fit_transform(x_train)

	model = MultinomialNB()
	model.fit(x_train, y_train)

	y_pred = model.predict(x_train)

	return(y_pred)

# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred,y_true):
	correct_pred = 0
	for i in range(len(y_pred)):
		if y_pred[i] == y_true[i]:
			correct_pred +=1
	return(correct_pred/len(y_pred))
