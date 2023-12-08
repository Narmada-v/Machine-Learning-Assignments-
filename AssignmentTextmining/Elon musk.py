# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:49:38 2023

@author: narma
"""

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
!pip install spacy
import spacy

from matplotlib.pyplot import imread
!pip install wordcloud

from wordcloud import WordCloud, STOPWORDS
%matplotlib inline
# load the dataset
tweets=pd.read_csv('C:/Users/narma/Downloads/Textmining/Elon_musk.csv',encoding='Latin-1')
tweets.drop(['Unnamed: 0'],inplace=True,axis=1)
tweets

#Text Preprocessing
tweets=[Text.strip() for Text in tweets.Text] # remove both the leading and the trailing characters
tweets=[Text for Text in tweets if Text] # removes empty strings, because they are considered in Python as False
tweets[0:10]

# Joining the list into one string/text
tweets_text=' '.join(tweets)
tweets_text

# remove Twitter username handles from a given twitter text. (Removes @usernames)
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(strip_handles=True)
tweets_tokens=tknzr.tokenize(tweets_text)
print(tweets_tokens)

# Again Joining the list into one string/text
tweets_tokens_text=' '.join(tweets_tokens)
tweets_tokens_text

# Remove Punctuations 
no_punc_text=tweets_tokens_text.translate(str.maketrans('','',string.punctuation))
no_punc_text

# remove https or url within text
import re
no_url_text=re.sub(r'http\S+', '', no_punc_text)
no_url_text

from nltk.tokenize import word_tokenize
text_tokens=word_tokenize(no_url_text)
print(text_tokens)

# Tokenization
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Tokens count
len(text_tokens)

# Remove Stopwords
from nltk.corpus import stopwords
my_stop_words=stopwords.words('english')

sw_list = ['\x92','rt','ye','yeah','haha','Yes','U0001F923','I']
my_stop_words.extend(sw_list)

no_stop_tokens=[word for word in text_tokens if not word in my_stop_words]
print(no_stop_tokens)

# Normalize the data
lower_words=[Text.lower() for Text in no_stop_tokens]
print(lower_words[100:200])

# Stemming (Optional)
from nltk.stem import PorterStemmer
ps=PorterStemmer()
stemmed_tokens=[ps.stem(word) for word in lower_words]
print(stemmed_tokens[100:200])

# Lemmatization
import spacy
print(spacy.util.is_package('en_core_web_sm'))

import spacy
print(spacy.__version__)

nlp=spacy.load('en_core_web_sm')
doc=nlp(' '.join(lower_words))
print(doc)

lemmas=[token.lemma_ for token in doc]
print(lemmas)

clean_tweets=' '.join(lemmas)
clean_tweets


#Feature Extaction
#1. Using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
tweetscv=cv.fit_transform(lemmas)
print(cv.vocabulary_)
import sklearn
print(sklearn.__version__)
feature_names = list(cv.vocabulary_.keys())
print(feature_names[100:200])
print(tweetscv.toarray()[100:200])
print(tweetscv.toarray().shape)


#2. CountVectorizer with N-grams (Bigrams & Trigrams)

cv_ngram_range = CountVectorizer(analyzer='word', ngram_range=(1, 3), max_features=100)
bow_matrix_ngram = cv_ngram_range.fit_transform(lemmas)

# Retrieve feature names
feature_names = cv_ngram_range.get_feature_names_out()
print(feature_names)
print(bow_matrix_ngram.toarray())


#3. TF-IDF Vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
tfidfv_ngram_max_features=TfidfVectorizer(norm='l2',analyzer='word',ngram_range=(1,3),max_features=500)
tfidf_matix_ngram=tfidfv_ngram_max_features.fit_transform(lemmas)
print(tfidfv_ngram_max_features.get_feature_names_out())
print(tfidf_matix_ngram.toarray())


#Generate Word Cloud
# Define a function to plot word cloud
def plot_cloud(wordcloud):
    plt.figure(figsize=(40,30))
    plt.imshow(wordcloud)
    plt.axis('off')
    
# Generate Word Cloud

STOPWORDS.add('pron')
STOPWORDS.add('rt')
STOPWORDS.add('yeah')
wordcloud=WordCloud(width=3000,height=2000,background_color='black',max_words=50,
                   colormap='Set1',stopwords=STOPWORDS).generate(clean_tweets)
plot_cloud(wordcloud)

#Named Entity Recognition (NER)
# Parts Of Speech (POS) Tagging
nlp=spacy.load('en_core_web_sm')

one_block=clean_tweets
doc_block=nlp(one_block)
spacy.displacy.render(doc_block,style='ent',jupyter=True)


for token in doc_block[100:200]:
    print(token,token.pos_) 
    
# Filtering the nouns and verbs only
nouns_verbs=[token.text for token in doc_block if token.pos_ in ('NOUN','VERB')]
print(nouns_verbs[100:200])

# Counting the noun & verb tokens
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

X=cv.fit_transform(nouns_verbs)
sum_words=X.sum(axis=0)

words_freq=[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]
words_freq=sorted(words_freq, key=lambda x: x[1], reverse=True)

wd_df=pd.DataFrame(words_freq)
wd_df.columns=['word','count']
wd_df[0:10] # viewing top ten results

# Visualizing results (Barchart for top 10 nouns + verbs)
wd_df[0:10].plot.bar(x='word',figsize=(12,8),title='Top 10 nouns and verbs');

from nltk import tokenize
sentences=tokenize.sent_tokenize(' '.join(tweets))
sentences

sent_df=pd.DataFrame(sentences,columns=['sentence'])
sent_df

# 1) Extract reviews of any product from ecommerce website like amazon
# 2) Perform emotion mining

import pandas as pd

df = pd.read_csv("C:/Users/narma/Downloads/amazon_reviews.csv")
df
df = df.drop(["asin","title","location_and_date","verified"],axis=1)
# SentimentIntensityAnalyzer
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

import string
# Remove punctuation from the text
translator = str.maketrans("", "", string.punctuation)
df['text'] = df['text'].apply(lambda text: text.translate(translator))


df["text"]

# Perform sentiment analysis on each text and create a new column for sentiment scores
df['sentiment_score'] = df['text'].apply(lambda text: sia.polarity_scores(text))


df['sentiment_score'] 
df

# Assume d1["sentiment_score"] contains the dictionary as mentioned above
df['neg_score'] = df['sentiment_score'].apply(lambda x: x['neg'])
df['neu_score'] = df['sentiment_score'].apply(lambda x: x['neu'])
df['pos_score'] = df['sentiment_score'].apply(lambda x: x['pos'])
df['compound_score'] = df['sentiment_score'].apply(lambda x: x['compound'])


df
# Classify sentiments into categories (positive, negative, neutral)
df['sentiment_label'] = df['sentiment_score'].apply(lambda scores: 'positive' if scores['compound'] > 0 else 'negative' if scores['compound'] < 0 else 'neutral')
df

# Split the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment_label'], test_size=0.3, random_state=42)
# Vectorize the text data
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Initialize the Logistic Regression model
from sklearn.linear_model import LogisticRegression
LE = LogisticRegression()

LE.fit(X_train_vect, y_train)

# Predictions on training set
y_train_pred = LE.predict(X_train_vect)

# Predictions on test set
y_test_pred = LE.predict(X_test_vect)

# Accuracy
from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)


print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)



