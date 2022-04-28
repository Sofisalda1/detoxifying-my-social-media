#%%
from config import TRACKING_URI, EXPERIMENT_NAME
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
# Initialization
import nltk 
nltk.download('wordnet')
nltk.download('omw-1.4')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from spellchecker import SpellChecker

import mlflow
WNlemma = nltk.WordNetLemmatizer()


def __get_data():
    #logger.info("Getting the data")
    #########################################
    # import data
    #########################################
    DATA_NAME = 'train_wikipedia_pre_clean'
    train = pd.read_csv("./data/" + DATA_NAME + ".csv")

    # cleaning data and preparing
    Y = train["toxic"][:50000]
    X = train["comment_text"][:50000]
    return X,Y, DATA_NAME



def run_baseline():
    X, Y,DATA_NAME = __get_data()
    logger.info("Calculating Baseline Model")
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


    with mlflow.start_run():
        mlflow.log_params({"Data": DATA_NAME})

        logger.info('Vectorizing and Lemmatizing')
        analyzer = CountVectorizer().build_analyzer()
        def lemmatize_word(doc):
            return (WNlemma.lemmatize(t) for t in analyzer(doc))

        lemm_vectorizer = CountVectorizer(min_df=15, analyzer=lemmatize_word)
        X_vect = lemm_vectorizer.fit_transform(X)
        mlflow.log_params({"Vectorizer": "Count", "Word_Summary": 'lemmatization'})
        lenvoc =  len(lemm_vectorizer.vocabulary_)
        logger.info(lenvoc)
        mlflow.log_params({'Vocabulary':lenvoc})


        # get toxic words

        model = LogisticRegression(max_iter=1500)
        model.fit(X_vect, Y)
        # get the feature names as numpy array
        feature_names = np.array(lemm_vectorizer.get_feature_names_out())
        # Sort the coefficients from the model (from lowest to highest values)
        sorted_coef_index = model.coef_[0].argsort()
        toxic_words_raw = feature_names[sorted_coef_index[:-300:-1]]
        spell = SpellChecker()
        correct_words = spell.known(toxic_words_raw)
        toxic_words = list(correct_words)
        logger.info('Transforming to array')
        X_array = X_vect.toarray()
        df_occurrence = pd.DataFrame(data=X_array,columns = lemm_vectorizer.get_feature_names_out())
        
        #words_in_corpus = lemm_vectorizer.get_feature_names_out()
        ## only keep those words that are in corpus
        #toxic_index = [i for i, curse_word in enumerate(toxic_words) if curse_word in words_in_corpus]
        #toxic_words = [toxic_words[i] for i in toxic_index]
        
        #logger.info('Calculating Score')
        pred = sum(df_occurrence[insult] for insult in toxic_words) > 0
        eval = roc_auc_score(Y, pred)
        #mlflow.log_metric(eval)
        mlflow.log_metric("ROC", eval)

if __name__ == "__main__":
    import logging

    logger = logging.getLogger()
    logging.basicConfig(format="%(asctime)s: %(message)s")
    logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
    logger.setLevel(logging.INFO)

    run_baseline()

