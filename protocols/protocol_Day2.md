
<span style="color:grey">

# Day 2, 20.04.2022

## Set Up

### Goals of the Week
* Explore the topic: what has been done before
* Explore the kaggle datasets
* Explore NLP methods
* Make baseline model
* EDA
* Fit first NLP models

### Progress
#### What has been done before
* fit CNN: https://towardsdatascience.com/toxic-comment-classification-using-lstm-and-lstm-cnn-db945d6b7986
    * Accuracy = 0.
* SVM and NB: https://www.turcomat.org/index.php/turkbilmat/article/download/2798/2425/5341#:~:text=Toxic%20comments%20were%20highest%20in,and%20threat%20in%20decreasing%20order.
* Preprocessing comments: 
    * https://www.kaggle.com/code/fizzbuzz/toxic-data-preprocessing/script
    * https://www.analyticsvidhya.com/blog/2020/04/beginners-guide-exploratory-data-analysis-text-data/
    * Spelling correction: https://towardsdatascience.com/essential-text-correction-process-for-nlp-tasks-f731a025fcc3

#### Datasets
* Project 1: wikipedia (https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
* Project 2: civil (https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
* Project 3: Multilingual (https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification)
* rename dataset according to data source (wikipedia, civil)
* wikipedia dataset has the downside that it's based on only one rater. ciliv data is based on multiple raters

#### Methods
* Summary of *Chapter 8: Social Media* (Vajjala et al.) below
#### Baseline Model
* Based on curse words/insults
    * Get list of English curse words and classify preprocessed comments without NLP

#### EDA

#### First NLP models
* AUC = 0.84


### Learnings
* Challenges of Social Media Text Data (In Vajjala et al).
    * **No grammar**: This departure from standard languages makes basic pre- processing steps like tokenization, POS tagging, and identification of sentence boundaries difficult. Modules specialized to work with SMTD are required to achieve these tasks.
    * **Nonstandard spelling**: In SMTD, words can have many spelling varia‐ tions. For an NLP system to work well, it needs to understand that all these words refer to the same word.
    * **Multilingual**: On social media, people often mix languages.
    * **Special Characters**:  one needs modules in the pre-processing pipelines to handle such non-textual entities.
    * **out of vocabulary (OOV) problem**
    * **condensed writing**
    * the tokenizer available in NLTK is designed to work with standard English language. Use one of the specialized tokenizer and use the one that gives the best output for your corpus and use case:
        * nltk.tokenize.TweetTokenizer
        * Twikenizer
        * Twokenizer by ARK at CMU
        * twokenize.
* Suggested **preprocessing** in Vajjala et al.:
    * Removing markup elements, e.g. /n (: soup = BeautifulSoup(markup))
    * Handling non-text data (Text = text.encode("utf-8"))
    * Handling apostrophes (manually)
    Handling emojis with Demoji module
    * Split-joined words
    * Removal of URLs
    * Nonstandard spellings
    ```{r}
    # for repeated letters
    def prune_multple_consecutive_same_char(tweet_text): '''
        yesssssssss  is converted to yes
        ssssssssssh is converted to ssh
        '''
        tweet_text = re.sub(r'(.)\1+', r'\1\1', tweet_text) 
        return tweet_text
    # from library
    from textblob import TextBlob
    data = "His sellection is bery antresting" output = TextBlob(data).correct() print(output)

    ```

        

### TO DO
* Get list of English curse words and classify preprocessed comments without NLP

### Ideas
* Make word cloud in presentation to display the most influential words. Here example based on count (and not influence):
```{r}
from wordcloud import WordCloud 
document_file_path = ‘./twitter_data.txt’ 
text_from_file = open(document_file_path).read()
stop_words = set(nltk.corpus.stopwords.words('english'))
word_tokens = twokenize(text_from_file)
filtered_sentence = [w for w in word_tokens if not w in stop_words] wl_space_split = " ".join(filtered_sentence)
my_wordcloud = WordCloud().generate(wl_space_split)
    plt.imshow(my_wordcloud)
    plt.axis("off")
    plt.show()
```