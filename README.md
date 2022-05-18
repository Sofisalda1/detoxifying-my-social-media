# Detoxify my Social Media
The anonymity on the internet allows people to use a more toxic tone when interacting on social media than in face-to-face contact. The use of insults and curse words, however, poses a risk on the mental health of social media users. In order to enable healthier communication on online platforms, we developed an algorithm that automatically detects toxic comments without cancelling the innocent use of curse words. Our future work will embed the algorithm in a web application to automatically delete toxic comments on your social media page.


## The Data
The data were extracted from kaggle () and and are governed by Wikipedia's CC-SA-3.0. The text data consist of Wikipedia comments which have been labeled by human raters for toxic behavior. The following columns are contained in the data

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

The `requirements.txt` file contains the libraries needed for deployment.. of model or dashboard .. thus no jupyter or other libs used during development.

The MLFLOW URI should **not be stored on git**, you have two options, to save it locally in the `.mlflow_uri` file:

```BASH
echo https://hudsju377cddpoevnjdkfnvpwovniewnipcdsnkvn.mlflow.neuefische.de > .mlflow_uri
```

This will create a local file where the uri is stored which will not be added on github (`.mlflow_uri` is in the `.gitignore` file). Alternatively you can export it as an environment variable with

```bash
export MLFLOW_URI=https://hudsju377cddpoevnjdkfnvpwovniewnipcdsnkvn.mlflow.neuefische.de
```

This links to your local mlflow, if you want to use a different one, then change the set uri.

The code in the [config.py](modeling/config.py) will try to read it locally and if the file doesn't exist will look in the env var.. IF that is not set the URI will be empty in your code.
