
<span style="color:grey">

# Day 3, 21.04.2022

## Set Up

### Goals of the Week
* Explore the topic: what has been done before
* Explore the kaggle datasets
* Explore NLP methods
* Make baseline model
* EDA
* Fit first NLP models

### Progress
#### Baseline Model
* how it works
    * make list of cursewords
    * make X_vectorize into array and then dictionary
    * find index in data of cursewords
* Problem: too many unique words. We need to to clean the data first.

#### Preprocessing
* Removing markup elements, e.g. /n (: soup = BeautifulSoup(markup))
* Handling non-text data (Text = text.encode("utf-8"))
* Handling apostrophes ('re' --> 'are')
    * We need to add to more apostrophe words (minuscule, capital letters)
    
* Handling emojis with Demoji module

* Split-joined words

* Removal of URLs

* Nonstandard spellings
