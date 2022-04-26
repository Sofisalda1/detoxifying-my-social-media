import pandas as pd
import re

train = pd.read_csv("./data/train_wikipedia_pre_clean.csv")
X = train['comment_text']

#test = pd.read_csv("./data/test_wikipedia_pre_clean.csv")
#X = test['comment_text']

X = X.apply(lambda x: re.sub(r'[0-9]+', '', x))

# Stock market tickers $GE
X = X .apply(lambda x: re.sub(r'\$\w*', '', x))

# remove hashtags
X  = X.apply(lambda x: re.sub(r'#', '', x))

from spellchecker import SpellChecker
def correct(x):
    spell = SpellChecker()
    misspelled = spell.unknown(x.split())
    for w in misspelled:
        return [w, spell.correction(w), list(spell.candidates(w))]


misspellings = []        
for i,text in enumerate(X):
    if correct(text):
        word, corrected_word, suggestions = correct(text)
        misspellings.append([i, word, corrected_word, suggestions])


df = pd.DataFrame(misspellings, columns = ['row', 'word', 'corrected', 'suggestions'])
df.to_csv('./data/train_spelling_correction.csv')
#df.to_csv('../data/test_spelling_correction.csv')
