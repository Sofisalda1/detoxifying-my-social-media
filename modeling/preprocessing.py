

#%%
#########################################
# # import modules
#########################################
import pandas as pd

import numpy as np
import warnings
warnings.filterwarnings('ignore')
from toolbox import *
WNlemma = nltk.WordNetLemmatizer()
stemmer = nltk.PorterStemmer()
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
    Y = train["toxic"]
    X = train["comment_text"]
    return X,Y, DATA_NAME


#########################################
# set parameters
#########################################

min_df = 20 # 1 20
stop_words = 'english' # {} 'english'
ngram = 1 # 1, 2
lowercase = True # True, False
tokenizer = tweettokenizer
vectorizer = CountVectorizer # TfidfVectorizer CountVectorizer
word_generalization = stemmer.stem # stemmer.stem WNlemma.lemmati
model = LogisticRegression()

#%%
#########################################
# preprocess and fit models
#########################################


def run_preprocessing():
    X, Y, DATA_NAME = __get_data()
    logger.info("Preprocessing Data")
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        mlflow.log_params({"Data": DATA_NAME})
        mlflow.log_params({"Min Occurrence": min_df,"Stop Words": stop_words,
                        "Ngram": ngram, "Lowercase": lowercase, "Tokenizer": tokenizer,
                        "Vectorizer": vectorizer, "Word_Summary": word_generalization, "ml_model": model})

        X_preproc, vect = nlp_preprocess(X, tokenizer,vectorizer, word_generalization, min_df, stop_words, ngram, lowercase)
        mlflow.log_params({"Vocabulary": len(vect.vocabulary_)})
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
        

