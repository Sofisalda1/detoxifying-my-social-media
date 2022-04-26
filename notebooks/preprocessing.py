#########################################
# # import modules
#########################################
import pandas as pd
import numpy as np
import re
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score
from statistics import mean
from sklearn.metrics import hamming_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

from sklearn.metrics import roc_auc_score, confusion_matrix
import statistics
from sklearn.metrics import recall_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from wordcloud import WordCloud
from collections import Counter

from sklearn.pipeline import Pipeline

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
#import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
from utils import *
import nltk 
from nltk.tokenize import TweetTokenizer


#########################################
# define functions
#########################################

stopwords_english = stopwords.words('english')

# Tokenizer
# create a function for the tweet tokenizer from NLTK
def tweettokenizer(text):
    tt = TweetTokenizer(preserve_case=True, strip_handles=True,
                            reduce_len=True)
    return tt.tokenize(text)
 

WNlemma = nltk.WordNetLemmatizer()
stemmer = nltk.PorterStemmer()
#nltk.download('wordnet')
#nltk.download('omw-1.4')

def nlp_preprocess(X, tokenizer, vectorizer, word_generalization, min_df=1, stop_words={}, ngram=1, lowercase=True):
    # Fit the CountVectorizer to the training data
    print('Vectorizing and Tokenizing...')
    analyzer = vectorizer(min_df=min_df, 
                            ngram_range=(1,ngram), stop_words=stop_words, 
                            tokenizer=tokenizer, 
                            lowercase=lowercase).build_analyzer()
    def change_word(doc):
        return (word_generalization(t) for t in analyzer(doc))
    vect = vectorizer(analyzer=change_word)
    X_vect = vect.fit_transform(X)
    print('Done!')
    return X_vect

def fit_nlp(X, Y):
    # Creating classifiers with default parameters initially.
    clf1 = MultinomialNB()
    clf2 = LogisticRegression()
    clf3 = LinearSVC()
    # Calculating the cross validation F1 and Recall score for our 3 baseline models.
    print('Fitting MultinomialNB...')
    methods1_cv = pd.DataFrame(cross_validation_score(clf1, X_vect, Y))
    print('Fitting LogisticRegression...')
    methods2_cv = pd.DataFrame(cross_validation_score(clf2, X_vect, Y))
    #methods3_cv = pd.DataFrame(cross_validation_score(clf3, X_vect, Y))

    # Creating a dataframe to show summary of results.
    print('Used Vectorizer: ' + str(tokenizer) + ' ' + str(vectorizer))
    print(' ')
    print(f"Used parameters: Minimim Occurrence: {min_df}, stop words: {stop_words}, ngram: {ngram}, lower case: {lowercase}") 
    methods_cv = pd.concat([methods1_cv, methods2_cv])
    methods_cv.columns = ['Model', 'Recall', 'F1', 'ROC']
    meth_cv = methods_cv.reset_index()
    print(meth_cv[['Model', 'Recall', 'F1', 'ROC']])

def cross_validation_score(classifier, X_train, y_train):
    '''
    Iterate though each label and return the cross validation F1 and Recall score 
    '''
    methods = []
    name = classifier.__class__.__name__.split('.')[-1]

    
    recall = cross_val_score(
        classifier, X_train, y_train, cv=10, scoring='recall')
    f1 = cross_val_score(classifier, X_train,
                        y_train, cv=10, scoring='f1')
    roc = cross_val_score(classifier, X_train,
                        y_train, cv=10, scoring='roc_auc')
    methods.append([name, recall.mean(), f1.mean(), roc.mean()])

    return methods

#########################################
# import data
#########################################
train = pd.read_csv("./data/train_wikipedia_pre_clean.csv")
test = pd.read_csv("./data/test_wikipedia_pre_clean.csv")
test_y = pd.read_csv("./data/test_labels_wikipedia_pre_clean.csv")

#########################################
# Additional Cleaning
#########################################
# non text
train["comment_text"]  = train["comment_text"] .apply(lambda x: x.encode("latin-1","ignore").decode('ISO-8859-1'))
test["comment_text"]  = test["comment_text"] .apply(lambda x: x.encode("latin-1","ignore").decode('ISO-8859-1'))

# numbers
train["comment_text"]  = train["comment_text"] .apply(lambda x: x.encode("ascii","ignore").decode('ISO-8859-1'))
test["comment_text"]  = test["comment_text"] .apply(lambda x: x.encode("ascii","ignore").decode('ISO-8859-1'))

# Stock market tickers $GE
train["comment_text"]  = train["comment_text"] .apply(lambda x: re.sub(r'\$\w*', '', x))
test["comment_text"]  = test["comment_text"] .apply(lambda x: re.sub(r'\$\w*', '', x))

# remove hashtags
train["comment_text"]  = train["comment_text"] .apply(lambda x: re.sub(r'#', '', x))
test["comment_text"]  = test["comment_text"] .apply(lambda x: re.sub(r'#', '', x))


#########################################
# define x and y
#########################################
X = train["comment_text"]
Y = train['toxic']
X_test = test["comment_text"]
#Y_test = test['toxic']



#########################################
# set parameters
#########################################
min_df = 20 # 1 20
stop_words = 'english' # {} 'english'
ngram = 1 # 1, 2
lowercase = False # True, False
tokenizer = tweettokenizer
vectorizer = CountVectorizer # TfidfVectorizer CountVectorizer
word_generalization = WNlemma.lemmatize # stemmer.stem WNlemma.lemmatize

#########################################
# preprocess and fit models
#########################################


# preprocess
X_vect = nlp_preprocess(X, tokenizer, vectorizer, word_generalization, min_df, stop_words, ngram, lowercase)
print(X_vect)

# fit models
fit_nlp(X_vect, Y)
