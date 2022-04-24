import pandas as pd
import demoji
import re
from textblob import TextBlob
import tqdm

def remove_doublequotation(panda_text_column):
    return panda_text_column.apply(lambda x: x.replace('"', ''))
Apostrophes_expansion = {
"i'm": "i am","I'm": "I am", "I'M": "I AM", 
"i'd": "i would", "I'd": "I would","I'D": "I WOULD",
"you're": "you are", "You're": "You are", "YOU'RE": "YOU ARE",
"he's": "he is","He's": "He is","HE'S": "HE IS",
"she's": "she is","She's": "She is","SHE'S": "SHE IS",
"it's":"it is", "It's":"It is", "IT'S":"IT IS",
"that's": "that is", "That's": "That is","THAT'S": "THAT IS",
"they're": "they are","They're": "They are","THEY'RE": "THEY ARE",
"we're": "we are","We're": "We are","WE'RE": "WE ARE",
"i've": "i have", "I've": "I have", "I'VE": "I HAVE",
"you've": "you have","You've": "You have","YOU'VE": "YOU HAVE",
"we've": "we have","We've": "We have","WE'VE": "WE HAVE",
"ain't": "are not","Ain't": "Are not","AIN'T": "ARE NOT",
"aren't": "are not", "Aren't": "Are not","AREN'T": "ARE NOT",
"isn't": "is not", "Isn't": "Is not", "ISN'T": "IS NOT",
"don't": "do not", "Don't": "Do not","DON'T": "DO NOT",
"doesn't": "does not", "Doesn't": "Does not", "DOESN'T": "DOES NOT",
"won't": "will not","Won't": "Will not", "WON'T": "WILL NOT",
"can't": "cannot","Can't": "Cannot", "CAN'T": "CANNOT",
"shouldn't": "should not","Shouldn't": "Should not","SHOULDN'T": "SHOULD NOT",
"wouldn't": "would not", "Wouldn't": "Would not", "WOULDN'T": "WOULD NOT",
"didn't": "did not", "Didn't": "Did not", "DIDN'T": "DID NOT",
"weren't":"were not","Weren't":"Were not", "WEREN'T":"WERE NOT", 
"wasn't":"was not","Wasn't":"Was not", "WASN'T":"WAS NOT", 
"there's": "there is", "There's": "There is", "THERE'S": "THERE IS",
"here's": "here is", "Here's": "Here is", "HERE'S": "HERE IS",
"let's": 'let us', "Let's": "Let us", "LET'S": "LET US"
} 
def interpolate_apostrophes_string(string):
    x = string.split(' ')
    return " ".join([Apostrophes_expansion[word] if word in Apostrophes_expansion else word for word in x])
def interpolate_apostrophes_column(panda_text_column):
    return panda_text_column.apply(lambda x: interpolate_apostrophes_string(x))

def correct_spelling(panda_text_column):
    return panda_text_column.apply(lambda x: TextBlob(x).correct())

def process_URLs_individual(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    return text
def process_URLs(panda_text_column):
    return panda_text_column.apply(lambda x: process_URLs_individual(x))
## emojis
# get dictionary of emojis
def get_emojis(panda_text_column):
    dict = {}
    for row in panda_text_column:
        dict.update(demoji.findall(row))
    return dict
def remove_emojis(df, column):
    df_copy = df.copy()
    for i,text in enumerate(df_copy[column]):
        dem = demoji.findall(text)
        if dem:
            for item in dem.keys():
                df_copy[column][i] = text.replace(item, '')
    return df_copy

def handle_nontext(panda_column):
    return panda_column.apply(lambda x: x.encode("latin-1","ignore").decode('ISO-8859-1'))

def remove_markups(panda_text_column):
    return panda_text_column.apply(lambda x: x.replace('\n', ' '))

def remove_patterns_string(text):
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
    return pattern.sub(r"\1", text)
def remove_repetitions(panda_text_column):
    return panda_text_column.apply(lambda x: remove_patterns_string(x))