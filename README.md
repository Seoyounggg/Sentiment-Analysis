# Sentiment Analysis

## 1. Dataset
> **English**
* Stanford Sentiment Treebank
- The corpus is based on the dataset introduced by Pang and Lee (2005) and consists of 11,855 single sentences extracted from moview reviews.  
- Each phrase is labeled as either negative, somewhat negative, neutral, somewhat positive or positive. (SST-5 or SST fine-grained)  
  - train: 8,544
  - dev: 1,101    
  - test: 2,210
- <https://paperswithcode.com/dataset/sst>

> **Korean**
* Naver Sentiment Movie Corpus
- The corpus is a movie review dataset in the Korean language. Reviews were scraped from Naver Movies.
- 200K reviews in total
  - ratings_train.txt: 150K reviews for training
  - ratings_test.txt: 50K reviews held out for testing
- <https://github.com/e9t/nsmc/>

## 2. Embedding
> **English**
* GloVe
* BERT (work in progress)

> **Korean**
* Character Embedding
* BERT (work in progress)

## 3. Results
> **English** 
 
**GloVe + CNN**
- Validation accuracy
<p align="center"><img src="https://user-images.githubusercontent.com/42035101/147531670-7620dbb5-0371-4aed-aab8-ac19dd144424.png" width="300"></p>  

- Test accuracy
<p align="center"><img src="https://user-images.githubusercontent.com/42035101/147531673-095b726e-c783-4707-b708-75a635ac581e.png" width="300"></p>

**GloVe + BiLSTM**
- Validation accuracy
<p align="center"><img src="https://user-images.githubusercontent.com/42035101/147531665-10806a25-47f9-4b3e-85a4-999fd24f4683.png" width="300"></p>  

- Test accuracy
<p align="center"><img src="https://user-images.githubusercontent.com/42035101/147531668-f0b1d4dc-af5d-4e5f-8b47-0be4eb269ef4.png" width="300"></p>


> **Korean**

**Character Embedding + CNN**
- Test accuracy
<p align="center"><img src="https://user-images.githubusercontent.com/42035101/147611302-26a50b0a-218a-4f56-aa2b-f4fe775bd347.png" width="300"></p>

**Character Embedding + BiLSTM**
- Test accuracy
<p align="center"><img src="https://user-images.githubusercontent.com/42035101/147611304-bbadeb57-8181-4fc4-ada9-8d4bc810b1c7.png" width="300"></p>
