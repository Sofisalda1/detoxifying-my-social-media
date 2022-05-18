# Detoxify my Social Media
The anonymity on the internet allows people to use a more toxic tone when interacting on social media than in face-to-face contact. The use of insults and curse words, however, poses a risk on the mental health of social media users. In order to enable healthier communication on online platforms, we developed an algorithm that automatically detects toxic comments without cancelling the innocent use of curse words. Our future work will embed the algorithm in a web application to automatically delete toxic comments on your social media page.


## The Data
The data were extracted from kaggle (https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data) and and are governed by Wikipedia's CC-SA-3.0. The text data consist of Wikipedia comments which have been labeled by human raters for toxic behavior. The following columns are contained in the data

comment_text: The text comment extracted from wikipedia.
toxic: the rating as toxic (1 for toxic and 0 for not toxic).

File descriptions
train.csv - the training set, contains comments with their binary labels
test.csv - the test set.

## Prediction
1. Explorative Data Analysis
2. Modelfitting
3. Dashboard



## Requirements:

- pyenv with Python: 3.9.8

### Setup

Use the requirements file in this repo to create a new environment.

```BASH
make setup

#or

pyenv local 3.9.8
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_dev.txt
```

The `requirements.txt` file contains the libraries needed for deployment of model and dashboard.
