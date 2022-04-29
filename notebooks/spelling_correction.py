import pandas as pd
import re
from tqdm import tqdm
from spellchecker import SpellChecker
from wordsegment import load, segment
load()
start=0
finish=1000
train = pd.read_csv("./data/train_wikipedia_pre_clean.csv")
X = train['comment_text'][start:finish]

#test = pd.read_csv("./data/test_wikipedia_pre_clean.csv")
#X = test['comment_text']

X = X.apply(lambda x: re.sub(r'[0-9]+', '', x))

# Stock market tickers $GE
X = X .apply(lambda x: re.sub(r'\$\w*', '', x))

# remove hashtags
X  = X.apply(lambda x: re.sub(r'#', '', x))


def correct(x):
    spell = SpellChecker()
    misspelled = spell.unknown(x.split())
    for w in misspelled:
        return [w, spell.correction(w), list(spell.candidates(w)), segment(w)]


misspellings = []     
print('Generating Misspelling Table...')
for i,text in enumerate(X):
    if correct(text):
        word, corrected_word, suggestions, suggestion_segmented = correct(text)
        misspellings.append([start+i, word, corrected_word, suggestions, suggestion_segmented])
        print(f'Row {i}: {word}')





df = pd.DataFrame(misspellings, columns = ['row', 'word', 'corrected', 'suggestions', 'segment suggestion'])

df.to_csv('./data/train_spelling_correction'+ str(start) + '.csv')
#df.to_csv('../data/test_spelling_correction.csv')
