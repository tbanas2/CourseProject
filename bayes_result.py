import sys
import nltk
import numpy as np
import pandas as pd
import pickle

from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
import os
from matplotlib import pyplot as plt
sys.path.append(".")
sys.path.append("..")

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


column_to_predict = "business_service"
classifier = "NB"  # Supported algorithms # "SVM" # "NB"
use_grid_search = False  # grid search is used to find hyperparameters. Searching for hyperparameters is time consuming
remove_stop_words = True  # removes stop words from processed text
stop_words_lang = 'english'  # used with 'remove_stop_words' and defines language of stop words collection
use_stemming = False  # word stemming using nltk
fit_prior = True  # if use_stemming == True then it should be set to False ?? double check
min_data_per_class = 1000 # used to determine number of samples required for each class.Classes with less than that will be excluded from the dataset. default value is 1

def train_bayes():

    dfTickets = pd.read_csv(
        './/all_tickets.csv',
        dtype=str
    )  

    text_columns = "body"  
    
    bytag = dfTickets.groupby(column_to_predict).aggregate(np.count_nonzero)
    tags = bytag[bytag.body > min_data_per_class].index
    dfTickets = dfTickets[dfTickets[column_to_predict].isin(tags)]
    labelData = dfTickets[column_to_predict]
    data = dfTickets[text_columns]

    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labelData, test_size=0.2
    )  

    if remove_stop_words:
        count_vect = CountVectorizer(stop_words=stop_words_lang)
    elif use_stemming:
        count_vect = StemmedCountVectorizer(stop_words=stop_words_lang)
    else:
        count_vect = CountVectorizer()

    # Fitting the training data into a data processing pipeline and eventually into the model itself

    text_clf = Pipeline([
        ('vect', count_vect),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB(fit_prior=fit_prior))
    ])
    text_clf = text_clf.fit(train_data, train_labels)


 
    # Score and evaluate model on test data using model without hyperparameter tuning
    predicted = text_clf.predict(test_data)
    prediction_acc = np.mean(predicted == test_labels)
    return test_labels, predicted


    
    

if __name__ == '__main__':
    train_bayes()
     
    