# Code_Samples

# Sentiment Classification and Movie Review Aspect Analysis

Sentiment classification, also referred to as opinion mining, is an approach that identifies the emotional tone behind a body of text. This is particularly useful for analyzing movie reviews, which is the focus of this NLP course project at NYU.

## Project Overview

The project aimed to go beyond just classifying sentiments but to advance the understanding of movie reviews by exploring movie aspects. Specifically, the project focuses on two separate sections:
1. **Classifying Multiple Sentiment Classes:** In this section, we classify the movie reviews into Positive, Extra_positive, Negative, and Extra_negative. The features used include word frequency, n-grams, and TF-IDF. The model Multinomial Naive Bayes generated the best performance with an F1 score of 0.58, outperforming the baseline’s F1 of 0.20.
   
2. **Segmenting Movie Review Sentences According to Various Aspects:** We segmented movie review sentences into five aspects: theater facilities, movie plots, actor & actress performances, movies’ special effects and scenes, and others. Control variable experiments using Cosine similarity and four machine learning algorithms on TF-IDF and Word2Vec features on differently modeled aspect target datasets were implemented. The algorithms tested include DummyClassifier (baseline), SVM, Random Forest, and KNN. For evaluation, we used the GPT-4 API to determine which aspect the movie review belongs to, with human annotations verifying the robustness of GPT-4’s ability to derive aspects. The K nearest neighbors model on Word2Vec embedding methods generated the best result with an F1 score of 0.83, significantly superseding the baseline F1 of 0.17.

## Conclusions

Overall, the results from both sections outperform the baseline model and fulfill the requirements for sentiment analysis and aspect segmentation. By extending the scope of sentiment classification research, this project can be used to provide more personalized movie review recommendations and has the potential to impact various applications beyond movie reviews, such as product reviews and social media analysis.


## Files

For a detailed report, please refer to the final project paper. \
For **Classifying Multiple Sentiment Classes:** section, please refer to sentiment_classification.ipynb\
For **Segmenting Movie Review Sentences According to Various Aspects:** section, please refer to sentence preprocessing & aspect-segmentation.ipynb.\
Thanks for your time to review the code and the project!
