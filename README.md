# Sentiment Analysis

## 1. Dataset
> English
* Stanford Sentiment Treebank
- The corpus is based on the dataset introduced by Pang and Lee (2005) and consists of 11,855 single sentences extracted from moview reviews.  
- Each phrase is labeled as either negative, somewhat negative, neutral, somewhat positive or positive. (SST-5 or SST fine-grained)  
  - train: 8,544
  - dev: 1,101    
  - test: 2,210
- <https://paperswithcode.com/dataset/sst>

> Korean
* Naver Sentiment Movie Corpus
- The corpus is a movie review dataset in the Korean language. Reviews were scraped from Naver Movies.
- 200K reviews in total
  - ratings_train.txt: 150K reviews for training
  - ratings_test.txt: 50K reviews held out for testing
- <https://github.com/e9t/nsmc/>

## 2. Embedding
> English
* GloVe
* BERT (work in progress)

> Korean
* Character Embedding
* BERT (work in progress)

## 3. Model
> English
* CNN
* BiLSTM

> Korean
* CNN
* BiGRU
