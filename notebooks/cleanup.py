# import packages
import pandas as pd
import demoji
import re
from cleanup_functions import *

##################################
# import data
##################################
INPUT_NAME='train_wikipedia'
#INPUT_NAME='train_civil'
df=pd.read_csv("../data/"+INPUT_NAME+ ".csv")

##################################
# cleanup
##################################
df['comment_text'] = remove_doublequotation(df['comment_text'])
df['comment_text'] = interpolate_apostrophes_column(df.comment_text)
df['comment_text'] = correct_spelling(df['comment_text'])
df['comment_text'] = process_URLs(df['comment_text'])
df = remove_emojis(df, 'comment_text')
df["comment_text"] = handle_nontext(df["comment_text"] )# then bytes make it diffcult to deal with data


##################################
# optional: show emojis in corpus
##################################
#x = get_emojis(df_wikipedia['comment_text'][:1000])

##################################
# save
##################################
df.to_csv(INPUT_NAME+'_clean.csv', index=False)