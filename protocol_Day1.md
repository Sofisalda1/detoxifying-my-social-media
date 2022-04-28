<span style="color:grey">

# Day 1, 19.04.2022

## Set Up

### Goals of the Week
* Explore the topic: what has been done before
* Explore the kaggle dataset
* Make baseline model
* EDA
* Fit first NLP models

### Learnings
* <span style="color:grey"> There are two kaggle projects:
    * 2018: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
    * 2020: http://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification
* About outcome variable: We use "toxic" column. Subclassifications (other columns) only match 95% with toxic classification.
* 10:90 ratio of toxic vs. non toxic comments.
* There are autographic errors.

### Open Questions

* Do we need correct autographic mistakes?
* Do we need to balance the data to scale up toxic comments to 50:50?


### To Do
* Look at the mismatches (-1) between toxic and subclassifications
* Collect literature
    * https://www.coursera.org/specializations/natural-language-processing?action=enroll&adgroupid=119269357576&adpostion=&campaignid=12490862811&creativeid=503940597773&device=c&devicemodel=&gclid=CjwKCAjwur-SBhB6EiwA5sKtjiRmEu7eWClxSgj6EBTMuMJVBi-UZqRWsjtdzQ9svGTMNnM3kr1aZxoC2CAQAvD_BwE&hide_mobile_promo&keyword=&matchtype=&network=g&utm_content=01-CatalogDSA-ML1-US&utm_medium=sem&utm_source=gg#howItWorks
    * Vajjala, S., Majumder, B., Gupta, A., & Surana, H. (2020). Practical Natural Language Processing: A Comprehensive Guide to Building Real-World NLP Systems. O'Reilly Media.