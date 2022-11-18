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

def get_bayes_result(column_to_predict = "business_service",text_columns = "body", remove_stop_words = True, stop_words_lang = 'english', use_stemming = False, fit_prior = True, min_data_per_class = 1000, file_path= './/all_tickets.csv' ):
    """Returns results of a Native Bayes classification when given a CSV.

    # Arguments
        column_to_predict = Name of column for classifying
        file_name = path to file
        text_columns = Name of column containing model input

    # Returns
        test_labels, predicted; the actual values of the labels in the test data set and what the model predicted
    """

    dfTickets = pd.read_csv(
        file_path,
        dtype=str
    )  
      
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

    text_clf = Pipeline([
        ('vect', count_vect),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB(fit_prior=fit_prior))
    ])
    text_clf = text_clf.fit(train_data, train_labels)

    predicted = text_clf.predict(test_data)
    prediction_acc = np.mean(predicted == test_labels)
    return test_labels, predicted


if __name__ == '__main__':
    get_bayes_result()
     
    