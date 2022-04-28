

#%%
#########################################
# # import modules
#########################################
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from toolbox import *
import mlflow


#%%
from config import TRACKING_URI, EXPERIMENT_NAME
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

#%%
def __get_data():
    DATA_NAME = 'train_wikipedia_pre_clean'
    train = pd.read_csv("./data/" + DATA_NAME + ".csv")

    # cleaning data and preparing
    Y = train["toxic"][:10000]
    X = train["comment_text"][:10000]
    return X,Y, DATA_NAME


#########################################
# set parameters
#########################################

min_df = 20 # 1 20
stop_words = 'english' # {} 'english'

WNlemma = nltk.WordNetLemmatizer()
stemmer = nltk.PorterStemmer()

ngram = 1 # 1, 2
lowercase = False # True, False
tokenizer = tweettokenizer
vectorizer = CountVectorizer # TfidfVectorizer CountVectorizer
word_generalization = WNlemma.lemmatize # stemmer.stem WNlemma.lemmatize

#%%
#########################################
# preprocess and fit models
#########################################

def run_preprocessing():
    min_df = 20 # 1 20
    stop_words = 'english' # {} 'english'
    WNlemma = nltk.WordNetLemmatizer()
    stemmer = nltk.PorterStemmer()
    ngram = 1 # 1, 2
    lowercase = False # True, False
    #tokenizer = tweettokenizer
    vectorizer = CountVectorizer # TfidfVectorizer CountVectorizer
    word_generalization = WNlemma.lemmatize # stemmer.stem WNlemma.lemmati
    model = LogisticRegression
    X, Y, DATA_NAME = __get_data()
    logger.info("Preprocessing Data")
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        mlflow.log_params({"Data": DATA_NAME})
        mlflow.log_params({"min_df": min_df,"stop_words": stop_words,
                        "ngram": ngram, "lowercase": lowercase, "tokenizer": tokenizer,
                        "vectorizer": vectorizer, "word_generalization": word_generalization, "ml_model": model})
        # preprocess
        X_preproc = nlp_preprocess(X, vectorizer, word_generalization, min_df, stop_words, ngram, lowercase)
        # fit models
        Precision, ROC = fit_nlp(X_preproc, Y, model)
        mlflow.log_metric("Precision", Precision)
        mlflow.log_metric("ROC", ROC)


if __name__ == "__main__":
    import logging

    logger = logging.getLogger()
    logging.basicConfig(format="%(asctime)s: %(message)s")
    logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
    logger.setLevel(logging.INFO)

    run_preprocessing()
        
