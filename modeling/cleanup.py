# import packages
import pandas as pd
#import demoji
import re
from cleanup_functions import *

from tqdm import tqdm
import warnings
import numpy as np

warnings.filterwarnings("ignore")
import string


##################################
# import data
##################################

#INPUT_NAME='test_wikipedia'
INPUT_NAME='train_civil'
df=pd.read_csv("./data/"+INPUT_NAME+ ".csv")
#df = df[:10000]
print(df.head())
##################################
# cleanup
##################################
print('(1/15) Removing double quotations')
for i in tqdm(range(1)):
    df['comment_text'] = remove_doublequotation(df['comment_text'])
print('(2/15) Interpolating Apostrophies')    
for i in tqdm(range(25)):
    df['comment_text'] = interpolate_apostrophes_column(df.comment_text)
print('(3/15) Removing Markups')    
for i in tqdm(range(20)):
     df['comment_text'] = remove_markups(df['comment_text'])
print('(4/15) Removing URLs')    
for i in tqdm(range(20)):
     df['comment_text'] = process_URLs(df['comment_text'])
print('(5/15) Removing repetitions')    
for i in tqdm(range(20)):
    df['comment_text'] = remove_repetitions(df['comment_text'])
print('(6/15) removing punctuation')    
for i in tqdm(range(10)):
    df['comment_text']  = df['comment_text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
print('(7/10) removing numbers')    
for i in tqdm(range(10)):
    df["comment_text"]  = df["comment_text"] .apply(lambda x: x.encode("ascii","ignore").decode('ISO-8859-1'))
print('(8/15) removing hashtags')    
for i in tqdm(range(10)):
    df["comment_text"]  = df["comment_text"] .apply(lambda x: re.sub(r'#', '', x))
print('(9/15) replacing abbreviations')    
for i in tqdm(range(10)):
    df["comment_text"]= df["comment_text"].apply(lambda x: replace_abbreviation(x))
print('(10/15) separating words')    
for i in tqdm(range(10)):
    df['comment_text'] = df['comment_text'].apply(lambda x: word_separator(x))
print('(11/15) Dealing with Emojis')    
for i in tqdm(range(100)):
    df = remove_emojis(df, 'comment_text')
print('(12/15) removing non-text')    
for i in tqdm(range(10)):
    df["comment_text"]  = df["comment_text"] .apply(lambda x: x.encode("latin-1","ignore").decode('ISO-8859-1'))
print('(13/15) Removing empty rows')
for i in tqdm(range(20)):
    df = df.replace('', np.nan).dropna(subset=['comment_text'])
# correct word
#print('(14/15) Correcting words')



##################################
# save
##################################
print('(15/15) saving')
#df.to_csv("./data/"+INPUT_NAME+'_clean.csv', index=False)
df.to_csv("./data/"+INPUT_NAME+'_clean.csv', index=True)

