from config import TRACKING_URI, EXPERIMENT_NAME
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
# Initialization
import nltk 
nltk.download('wordnet')
nltk.download('omw-1.4')
from sklearn.feature_extraction.text import CountVectorizer
import mlflow
WNlemma = nltk.WordNetLemmatizer()
stemmer = nltk.PorterStemmer()


def __get_data():
    logger.info("Getting the data")
    #########################################
    # import data
    #########################################
    train = pd.read_csv("./data/train_wikipedia.csv")

    # cleaning data and preparing
    Y = train["toxic"][:1000]
    X = train["comment_text"][:1000]
    return X,Y



def run_baseline():
    X, Y = __get_data()
    logger.info("Calculating Baseline Model")
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    toxic_words = ['bitch', 'fuck', 'shit', 'piss', 'dick', 'motherfucker', 'ass', 'asshole', 'bastard', 'damn', 'cunt', 'faggot', 'slut', 'whore']

    with mlflow.start_run():
        logger.info('Vectorizing and Lemmatizing')
        WNlemma = nltk.WordNetLemmatizer()
        analyzer = CountVectorizer().build_analyzer()
        def lemmatize_word(doc):
            return (WNlemma.lemmatize(t) for t in analyzer(doc))

        lemm_vectorizer = CountVectorizer(min_df=15, analyzer=lemmatize_word)
        X_vect = lemm_vectorizer.fit_transform(X)
        logger.info(len(lemm_vectorizer.vocabulary_))
        logger.info('Transforming to array')
        X_array = X_vect.toarray()
        df_occurrence = pd.DataFrame(data=X_array,columns = lemm_vectorizer.get_feature_names_out())
        words_in_corpus = lemm_vectorizer.get_feature_names_out()
        # only keep those words that are in corpus
        toxic_index = [i for i, curse_word in enumerate(toxic_words) if curse_word in words_in_corpus]
        toxic_words = [toxic_words[i] for i in toxic_index]
        
        logger.info('Calculating Score')
        pred = sum(df_occurrence[insult] for insult in toxic_words) > 0
        eval = roc_auc_score(X, pred)
        mlflow.log_metric("ROC-AUC: ", eval)

if __name__ == "__main__":
    import logging

    logger = logging.getLogger()
    logging.basicConfig(format="%(asctime)s: %(message)s")
    logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
    logger.setLevel(logging.INFO)

    run_baseline()
