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
    DATA_NAME = 'wikipedia_pre_clean'
    train = pd.read_csv("./data/train_" + DATA_NAME + ".csv")
    test = pd.read_csv("./data/test_" + DATA_NAME + ".csv")
    # cleaning data and preparing
    Y = train["toxic"]
    X = train["comment_text"]
    Y_test = test["toxic"]
    X_test= test["comment_text"]
    return X,Y, X_test, Y_test, DATA_NAME



def run_baseline():
    X, Y, X_test, Y_test, DATA_NAME = __get_data()
    logger.info("Calculating Baseline Model")
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


    with mlflow.start_run():
        mlflow.log_params({"Data": DATA_NAME})

        logger.info('Vectorizing and Lemmatizing')
        analyzer = CountVectorizer().build_analyzer()
        def lemmatize_word(doc):
            return (WNlemma.lemmatize(t) for t in analyzer(doc))

        vect = CountVectorizer(min_df=20, analyzer=lemmatize_word)
        X_vect = vect.fit_transform(X)
        X_vect_test = vect.transform(X_test)

        mlflow.log_params({"Vectorizer": "Count", "Word_Summary": 'lemmatization'})
        lenvoc =  len(vect.vocabulary_)
        logger.info(lenvoc)
        mlflow.log_params({'Vocabulary':lenvoc})
        logger.info('Transforming to array')
        X_array = X_vect.toarray()
        #X_array_t = X_vect_test.toarray()

        df_occurrence = pd.DataFrame(data=X_array,columns = vect.get_feature_names_out())
        #df_occurrence_test = pd.DataFrame(data=X_array_t,columns = vect.get_feature_names_out())
        

        # get toxic words
        logger.info('Calculating Score')
        toxic_words = ['fuck','fucking','shit','stupid','suck','bitch','ass','gay','dick','idiot','asshole',
                        'hell','cunt','faggot','hate','penis','sucks','cock','fag','crap','dumb','fat','nigger',
                        'bastard','bullshit','damn','moron','fucker','loser','idiots','fuckin','nazi','motherfucker','pussy','jerk','retard']
        words_in_corpus = vect.get_feature_names_out()
        df_baseline_voc = pd.DataFrame(words_in_corpus, columns=['word'])
        df_baseline_voc.to_csv('./data/baselinvoc.csv')
        # # only keep those words that are in corpus
        toxic_index = [i for i, curse_word in enumerate(toxic_words) if curse_word in words_in_corpus]
        toxic_words = [toxic_words[i] for i in toxic_index]

        pred_train = sum(df_occurrence[insult] for insult in toxic_words) > 0
        ROC_train = roc_auc_score(Y, pred_train)
        #pred_test = sum(df_occurrence_test[insult] for insult in toxic_words) > 0
        #ROC_test = roc_auc_score(Y_test, pred_test)
        mlflow.log_metric("ROC Train", ROC_train)
        #mlflow.log_metric("ROC Test", ROC_test)        

if __name__ == "__main__":
    import logging

    logger = logging.getLogger()
    logging.basicConfig(format="%(asctime)s: %(message)s")
    #logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
    logger.setLevel(logging.INFO)

    run_baseline()


# %%
