

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
    DATA_NAME = 'wikipedia_pre_clean'
    train = pd.read_csv("./data/train_" + DATA_NAME + ".csv")
    test = pd.read_csv("./data/test_" + DATA_NAME + ".csv")
    # cleaning data and preparing
    Y = train["toxic"]
    X = train["comment_text"]
    Y_test = test["toxic"]
    X_test= test["comment_text"]
    return X,Y, X_test, Y_test, DATA_NAME


#########################################
# set parameters
#########################################

min_df = 6 # 1 20
stop_words = 'english' # {} 'english'
ngram = 1 # 1, 2
lowercase = False # True, False
tokenizer = nltk.word_tokenize #tweettokenizer
vectorizer = TfidfVectorizer # TfidfVectorizer CountVectorizer
word_generalization = WNlemma.lemmatize # stemmer.stem WNlemma.lemmati
model = LogisticRegression()

#%%
#########################################
# preprocess and fit models
#########################################


def run_preprocessing():
    X, Y, X_test, Y_test, DATA_NAME = __get_data()
    logger.info("Preprocessing Data")
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        mlflow.log_params({"Data": DATA_NAME})
        mlflow.log_params({"Min Occurrence": min_df,"Stop Words": stop_words,
                        "Ngram": ngram, "Lowercase": lowercase, "Tokenizer": tokenizer,
                        "Vectorizer": vectorizer, "Word_Summary": word_generalization, "ml_model": model})

        X_preproc, vect = nlp_preprocess(X, tokenizer,vectorizer, word_generalization, min_df, stop_words, ngram, lowercase)
        X_test_preproc = vect.transform(X_test)
        mlflow.log_params({"Vocabulary": len(vect.vocabulary_)})
        eval = fit_nlp(X_preproc, Y, X_test_preproc, Y_test, model)
        mlflow.log_metric("Precision Train", eval.Precision_Train[0])
        mlflow.log_metric("ROC Train", eval.ROC_Train[0])
        mlflow.log_metric("Precision Test", eval.Precision_Test[0])
        mlflow.log_metric("ROC Test", eval.ROC_Test[0])


if __name__ == "__main__":
    import logging

    logger = logging.getLogger()
    logging.basicConfig(format="%(asctime)s: %(message)s")
    logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
    logger.setLevel(logging.INFO)

    run_preprocessing()
        

