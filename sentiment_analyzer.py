from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
import random
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import re
import time
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
################################################

#read data 
sms = pd.read_table('data/data.tsv')

stop = set(stopwords.words('english')) #select words not to include in vocabulary

X = sms.review
y = sms.sentiment

#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z]+') 

#split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1) 

#vectorizer to extract features-words or vocabulary from the data
vect = CountVectorizer(stop_words = stop, ngram_range = (1,2), tokenizer = token.tokenize)

#this extracts all the features or words from the train_data
vect.fit(X_train)

#This returns a document-term matrix which is a scipy-csr. This Format compressed the data to a large extent and increases the computation speed
X_train_dtm = vect.transform(X_train)

print(len(vect.get_feature_names()))

#convert the test-set into the same format
X_test_dtm = vect.transform(X_test)

# forest = RandomForestClassifier(n_estimators = 100) 
# forest = forest.fit( X_train_dtm, y_train )


logreg = LogisticRegression()

logreg.fit(X_train_dtm, y_train)

y_pred_class = logreg.predict(X_test_dtm)

acc = metrics.accuracy_score(y_test, y_pred_class)
print(acc)

confusion_mat = metrics.confusion_matrix(y_test, y_pred_class)
print(confusion_mat)


pred = ["An empty headed horror with nothing to recommend"]

#convert the example text into a format acceptable by the classifier,i.e, The same thing we did with our training and test-set
pred = vect.transform(pred)


prediction = logreg.predict(pred)
print(prediction)
