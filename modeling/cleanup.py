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
INPUT_NAME='train_wikipedia'
#INPUT_NAME='train_civil'
df=pd.read_csv("./data/"+INPUT_NAME+ "_pre_clean.csv")


##################################
# cleanup
##################################
''' print('(1/8) Removing double quotations')
for i in tqdm(range(1)):
    df['comment_text'] = remove_doublequotation(df['comment_text'])
print('(2/8) Interpolating Apostrophies')    
for i in tqdm(range(25)):
    df['comment_text'] = interpolate_apostrophes_column(df.comment_text)
print('(3/8) Removing Markups')    
for i in tqdm(range(20)):
    df['comment_text'] = remove_markups(df['comment_text'])
print('(4/8) Removing URLs')    
for i in tqdm(range(20)):
    df['comment_text'] = process_URLs(df['comment_text'])
print('(5/8) Removing repetitions')    
for i in tqdm(range(20)):
    df['comment_text'] = remove_repetitions(df['comment_text'])
print('(6/8) Dealing with Emojis')    
for i in tqdm(range(100)):
   df = remove_emojis(df, 'comment_text')
print('(7/8) removing punctuation')    
for i in tqdm(range(10)):
    df['comment_text']  = df['comment_text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
print('(8/8) Removing empty rows')
for i in tqdm(range(20)):
     df = df.replace('', np.nan).dropna(subset=['comment_text'])

# remove non-ascii
# words = [word.encode('ascii', 'ignore').decode('ascii') for word in words]

# reset index


# remove white space
# sentence.strip()
# print('(6/7) Correct Spelling')    
# for i in tqdm(range(1)):
#     df['comment_text'] = correct_spelling(df['comment_text'])
# #print('(7/7) Handling remaining non-text')    
'''


#########################################
# Additional Cleaning
#########################################
# non text
df["comment_text"]  = df["comment_text"] .apply(lambda x: x.encode("latin-1","ignore").decode('ISO-8859-1'))

# numbers
df["comment_text"]  = df["comment_text"] .apply(lambda x: x.encode("ascii","ignore").decode('ISO-8859-1'))

# Stock market tickers $GE
df["comment_text"]  = df["comment_text"] .apply(lambda x: re.sub(r'\$\w*', '', x))

# remove hashtags
df["comment_text"]  = df["comment_text"] .apply(lambda x: re.sub(r'#', '', x))

#nas
print('(8/8) Removing empty rows')
for i in tqdm(range(20)):
     df = df.replace('', np.nan).dropna(subset=['comment_text'])

##################################
# optional: show emojis in corpus
##################################
#x = get_emojis(df_wikipedia['comment_text'][:1000])



##################################
# save
##################################
#df.to_csv("./data/"+INPUT_NAME+'_clean.csv', index=False)
df.to_csv("./data/"+INPUT_NAME+'_pre_clean.csv', index=False)