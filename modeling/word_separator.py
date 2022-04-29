import pandas as pd
import re
from tqdm import tqdm
from spellchecker import SpellChecker
from wordsegment import load, segment
spell = SpellChecker()




load()

df = pd.read_csv("./data/train_wikipedia_pre_clean.csv")
df = df[:10]
def word_separator(text):
    misspelled = spell.unknown(text.split())
    if misspelled: 
        return " ".join([" ".join(segment(word)) for word in text.split()])
    else: 
        return text
for i in tqdm(range(100)):
    df['comment_text'] = df['comment_text'].apply(lambda x: word_separator(x))

df.to_csv('./data/train_wikipedia_wordsep.csv')