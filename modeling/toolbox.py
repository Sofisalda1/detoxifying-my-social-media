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
from sklearn.metrics import precision_score, roc_auc_score

import warnings
warnings.filterwarnings('ignore')



def nlp_preprocess(X, tokenizer, vectorizer, word_generalization, min_df=1, stop_words={}, ngram=1, lowercase=True):
    # Fit the CountVectorizer to the training data
    print('Vectorizing and Tokenizing...')
    analyzer = vectorizer().build_analyzer()
    def change_word(doc):
        return (word_generalization(t) for t in analyzer(doc))
    vect = vectorizer(analyzer=change_word, tokenizer=tokenizer, min_df=min_df, 
                            ngram_range=(ngram,ngram), stop_words=stop_words, 
                            lowercase=lowercase)
    X_vect = vect.fit_transform(X)
    print('Done!')
    return X_vect, vect

def fit_nlp(X, Y, X_test, Y_test, ml_model):
    results = pd.DataFrame(columns = ['Precision_Train', 'ROC_Train', 'Precision_Test', 'ROC_Test'], index=range(1))

    # Creating classifiers with default parameters initially.
    clf = ml_model
    # Calculating the cross validation F1 and Recall score for our 3 baseline models.
    print('Fitting Model...')
    # get performance of train data
    ml_model.fit(X, Y)
    predicted_train = ml_model.predict(X)
    results.Precision_Train = precision_score(Y,predicted_train, average="weighted")
    results.ROC_Train = roc_auc_score(Y,predicted_train,average="weighted")
    # get performance of test data
    predict_df = pd.DataFrame()
    predicted_test = ml_model.predict(X_test)
    predict_df["toxic"] = predicted_test
    results.Precision_Test = precision_score(Y_test[Y_test != -1],
                        predicted_test[Y_test != -1],
                        average="weighted")
    results.ROC_Test = roc_auc_score(Y_test[Y_test != -1],
                        predicted_test[Y_test != -1],
                average="weighted")
    return results


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