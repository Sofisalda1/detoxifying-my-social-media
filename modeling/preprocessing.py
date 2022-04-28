

#%%
#########################################
# # import modules
#########################################
import pandas as pd
#import numpy as np
#import statistics

#import re
#import string
#from timeit import default_timer as timer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
#from sklearn.svm import LinearSVC
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestClassifier

#from sklearn.pipeline import Pipeline
#from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, hamming_loss, fbeta_score
from sklearn.model_selection import cross_val_score#, StratifiedKFold, GridSearchCV, ShuffleSplit, learning_curve
from sklearn.feature_extraction.text import CountVectorizer#, TfidfVectorizer

import nltk 
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
#from nltk.stem.wordnet import WordNetLemmatizer

#from wordcloud import WordCloud
#from collections import Counter

import warnings
warnings.filterwarnings('ignore')
from utils import *
import mlflow


#%%
from config import TRACKING_URI, EXPERIMENT_NAME
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

#%%
#########################################
# define functions
#########################################
nltk.download('stopwords')
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
    #clf3 = LinearSVC()

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

#%%
#########################################
# import data
#########################################
train = pd.read_csv("./data/train_wikipedia_pre_clean.csv")
test = pd.read_csv("./data/test_wikipedia_pre_clean.csv")
test_y = pd.read_csv("./data/test_labels_wikipedia_pre_clean.csv")


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

#%%
#########################################
# preprocess and fit models
#########################################

# preprocess
X_vect = nlp_preprocess(X, tokenizer, vectorizer, word_generalization, min_df, stop_words, ngram, lowercase)
print(X_vect)

# fit models
fit_nlp(X_vect, Y)
