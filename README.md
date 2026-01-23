# Reddit NLP: Cryptocurrency vs. Stock Post Classification

## Table of Contents
1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Data Collection and Description](#data-collection-and-description)
4. [Methodology](#methodology)
5. [Results and Discussion](#results-and-discussion)
6. [Conclusions](#conclusions)
7. [Limitations and Future Work](#limitations-and-future-work)
8. [References](#references)

## Introduction

Reddit, a popular social news and discussion platform, hosts a variety of communities (subreddits) dedicated to different topics. Among these, r/wallstreetbets and r/CryptoMoonShots are two prominent subreddits focused on stock trading and cryptocurrency investments, respectively. This project aims to leverage Natural Language Processing (NLP) techniques to distinguish between posts from these two subreddits, providing insights into the linguistic patterns and content characteristics of cryptocurrency and stock-related discussions.

## Problem Statement

The primary objective of this study is to develop a robust binary classification model capable of accurately distinguishing between cryptocurrency-based posts from r/CryptoMoonShots and stock-based posts from r/wallstreetbets. By achieving this, we aim to:

1. Demonstrate the effectiveness of NLP techniques in categorizing financial discussion topics.
2. Provide a tool for automatic categorization of financial posts, which could be valuable for content moderation, trend analysis, or personalized content recommendations.
3. Identify key linguistic features that differentiate cryptocurrency discussions from stock market discussions.

## Data Collection and Description

Data was collected using the Python Reddit API Wrapper (PRAW), focusing on the following subreddits:

1. r/wallstreetbets: [https://www.reddit.com/r/wallstreetbets](https://www.reddit.com/r/wallstreetbets)
2. r/CryptoMoonShots: [https://www.reddit.com/r/CryptoMoonShots](https://www.reddit.com/r/CryptoMoonShots)

The dataset comprises posts from various categories (top, new, controversial) for each subreddit. Key features include:

| Feature         | Type     | Description                                      |
|-----------------|----------|--------------------------------------------------|
| id              | object   | Unique Reddit post ID                            |
| datetime        | object   | Post timestamp                                   |
| title           | object   | Post title                                       |
| text            | object   | Post content                                     |
| score           | int64    | Net upvotes (upvotes - downvotes)                |
| upvote_ratio    | float64  | Ratio of upvotes to total votes                  |
| url             | object   | URL of the Reddit post                           |
| subreddit       | int64    | Binary label (0: CryptoMoonShots, 1: wallstreetbets) |
| has_text        | bool     | Indicates presence of text content               |
| title_len       | int64    | Character length of the title                    |
| text_len        | int64    | Character length of the text content             |

## Methodology

Our approach to developing a classification model for Reddit posts involved the following steps:

1. **Data Collection**: Utilized PRAW to scrape posts from r/wallstreetbets and r/CryptoMoonShots.

2. **Data Preprocessing**:
   - Cleaned and combined data from different post categories.
   - Handled missing values and outliers.
   - Binarized the 'subreddit' column for model prediction.

3. **Text Processing**:
   - Tokenization
   - Lemmatization
   - Lowercasing
   - Removal of punctuation and stop words

4. **Feature Engineering**:
   - Created 'has_text', 'title_len', and 'text_len' features.

5. **Vectorization**:
   - Implemented both CountVectorizer (CVEC) and TF-IDF vectorization.

6. **Model Development**:
   - Trained and compared multiple classification models:
     - Multinomial Naive Bayes
     - Logistic Regression
     - Random Forest

7. **Hyperparameter Tuning**:
   - Utilized GridSearchCV for optimizing model parameters.

8. **Model Evaluation**:
   - Assessed model performance using accuracy, precision, recall, and F1-score.
   - Implemented cross-validation to ensure model generalization.

9. **Ensemble Methods**:
   - Explored ensemble techniques combining Naive Bayes and Logistic Regression.

## Results and Discussion

Our analysis yielded several key insights:

1. The Multinomial Naive Bayes model demonstrated superior performance, achieving an accuracy of just under 93% in predicting the subreddit of origin for a given post.

2. Cross-validation confirmed the model's robustness and generalizability.

3. An ensemble model combining Naive Bayes and Logistic Regression slightly improved accuracy to just above 93%.

4. Key differentiating features between cryptocurrency and stock posts were identified, providing insights into the unique language and topics of each community.

These findings demonstrate the effectiveness of NLP techniques in distinguishing between cryptocurrency and stock-related discussions on Reddit, highlighting the potential for automated content categorization in financial forums.

## Conclusions

This study successfully developed a high-accuracy classification model for distinguishing between cryptocurrency and stock-related posts on Reddit. The Multinomial Naive Bayes model, along with the ensemble approach, proved particularly effective in capturing the linguistic nuances of these two financial communities.

Our results suggest that there are indeed distinct language patterns and topics that characterize discussions in r/CryptoMoonShots and r/wallstreetbets. This insight could be valuable for content moderators, financial analysts, and researchers studying online financial communities.

## Limitations and Future Work

While our model demonstrates strong predictive performance, several areas for improvement and future research have been identified:

1. **Temporal Analysis**: Investigate how language patterns in these subreddits change over time, particularly in response to market events.

2. **Multi-class Classification**: Extend the model to classify posts into more granular categories (e.g., specific cryptocurrencies or stock sectors).

3. **Sentiment Analysis**: Incorporate sentiment analysis to understand the emotional tone of posts and its relationship to the subject matter.

4. **Deep Learning Approaches**: Explore the use of neural networks, particularly Convolutional Neural Networks (CNNs) or Transformers, for potentially higher classification accuracy.

5. **Feature Importance Analysis**: Conduct a more in-depth analysis of the most important features (words or phrases) that distinguish between cryptocurrency and stock posts.

6. **Cross-platform Analysis**: Extend the study to include data from other social media platforms to compare discussion patterns across different online communities.

7. **Real-time Classification**: Develop a system for real-time classification of new posts, which could be useful for live monitoring of financial discussions.

8. **Ethical Considerations**: Address potential biases in the data and model, and consider the ethical implications of automated content classification in financial discussions.

By addressing these limitations and exploring these avenues for future work, we can further improve the accuracy and applicability of our classification model, potentially extending its use to other domains of online financial discourse analysis.

## References

1. Reddit API Documentation. (n.d.). Retrieved from [https://www.reddit.com/dev/api/](https://www.reddit.com/dev/api/)

2. Python Reddit API Wrapper (PRAW). (n.d.). Retrieved from [https://praw.readthedocs.io/](https://praw.readthedocs.io/)

3. Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python. O'Reilly Media.

4. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

5. Wallstreetbets subreddit. (n.d.). Retrieved from [https://www.reddit.com/r/wallstreetbets](https://www.reddit.com/r/wallstreetbets)

6. CryptoMoonShots subreddit. (n.d.). Retrieved from [https://www.reddit.com/r/CryptoMoonShots](https://www.reddit.com/r/CryptoMoonShots)

7. Zzeniale. (n.d.). Subreddit-classification. GitHub. Retrieved from [https://github.com/zzeniale/Subreddit-classification](https://github.com/zzeniale/Subreddit-classification)

