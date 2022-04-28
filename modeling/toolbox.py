import pandas as pd
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import nltk 
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import warnings
warnings.filterwarnings('ignore')



def nlp_preprocess(X, vectorizer, word_generalization, min_df=1, stop_words={}, ngram=1, lowercase=True):
    # Fit the CountVectorizer to the training data
    print('Vectorizing and Tokenizing...')
    analyzer = vectorizer(min_df=min_df, 
                            ngram_range=(1,ngram), stop_words=stop_words, 
                            lowercase=lowercase).build_analyzer()
    def change_word(doc):
        return (word_generalization(t) for t in analyzer(doc))
    vect = vectorizer(analyzer=change_word)
    X_vect = vect.fit_transform(X)
    print('Done!')
    return X_vect, vect

def fit_nlp(X, Y, ml_model):

    # Creating classifiers with default parameters initially.
    clf = ml_model()


    # Calculating the cross validation F1 and Recall score for our 3 baseline models.
    print('Fitting Model...')
    methods_cv = pd.DataFrame(cross_validation_score(clf, X, Y))

    # Creating a dataframe to show summary of results.
    methods_cv.columns = ['Model', 'Precision', 'ROC']
    meth_cv = methods_cv.reset_index()
    return methods_cv.Precision, methods_cv.ROC


def cross_validation_score(classifier, X_train, y_train):
    '''
    Iterate though each label and return the cross validation F1 and Recall score 
    '''
    methods = []
    name = classifier.__class__.__name__.split('.')[-1]
    precision = cross_val_score(classifier, X_train,
                        y_train, cv=10, scoring='precision')
    roc = cross_val_score(classifier, X_train,
                        y_train, cv=10, scoring='roc_auc')
    methods.append([name, precision.mean(), roc.mean()])

    return methods



# Tokenizer
# create a function for the tweet tokenizer from NLTK
def tweettokenizer(text):
    tt = TweetTokenizer(preserve_case=True, strip_handles=True,
                            reduce_len=True)
    return tt.tokenize(text)