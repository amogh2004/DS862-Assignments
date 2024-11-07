#!/usr/bin/env python
# coding: utf-8

# # DS 862 - ASSIGNMENT 5
# ## AMOGH RANGANATHAIAH (aranganathaiah@sfsu.edu)
# ## EKTA SINGH (esingh@sfsu.edu)
# 
# For this assignment, we will be using [yelp dataset](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences), that contains Yelp reviews and the labeled sentiments.

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


# Load the data
yelp_data = pd.read_csv('yelp_labelled.txt', sep = "\t", names =['text','sentiment'])
yelp_data


# In[3]:


# Drop missing values
yelp_data.dropna(inplace=True)


# In[4]:


# from sklearn.model_selection import train_test_split

# # Separate the data into features and target variable
# X = yelp_data['text']
# y = yelp_data['sentiment']

# # Split the data into training and testing sets (80% for training, 20% for testing)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


import nltk
from nltk.corpus import stopwords
import re

# Download stopwords from nltk if not already done
# nltk.download('stopwords')

# Initialize stopwords
stop_words = set(stopwords.words('english'))

# Function to remove stop words from a given text
def remove_stopwords(text):
    words = re.findall(r'\b\w+\b', text.lower())  # Tokenize and convert to lowercase
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Apply the function to remove stop words from the text data
yelp_data['text'] = yelp_data['text'].apply(remove_stopwords)

# Now separate the data into features and target variable
X = yelp_data['text']
y = yelp_data['sentiment']

# Split the data into training and testing sets (80% for training, 20% for testing)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

# Feature Extraction using Bag-of-Words
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Build Naive Bayes Classifier
naive_bayes = MultinomialNB()

# Hyperparameter Tuning using GridSearchCV
# Define the hyperparameter grid
param_grid = {'alpha': [0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]}  # 'alpha' is the smoothing parameter

# Use GridSearchCV to find the best 'alpha'
grid_search = GridSearchCV(naive_bayes, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_bow, y_train)

# Get the best model
best_naive_bayes = grid_search.best_estimator_

# Evaluate the performance on the test set
y_pred = best_naive_bayes.predict(X_test_bow)
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

print("Best Hyperparameters:", grid_search.best_params_)
print("Accuracy on Test Set:", accuracy)
print("Classification Report:\n", classification_report_result)


# ### Observation for Bag-of-Words for Feature Extraction
# 
# 1. The optimal smoothing parameter alpha for the Multinomial Naive Bayes model was found to be 10.
# 2. The model achieved an overall accuracy of 75.5% on the test data.
# 3. Class 0 (Negative Sentiment):
#         a. Precision (0.73): Of all reviews predicted as negative, 73% were actually negative. The model makes some false positive errors when identifying negative reviews.
#     b. Recall (0.78): Of all actual negative reviews, the model correctly identified 78%. This indicates that the model performs well in capturing most negative reviews.
#     c. F1-Score (0.75): The harmonic mean of precision and recall, indicating a good balance between both metrics.
# 4. Class 1 (Positive Sentiment): </br>
#     a. Precision (0.78): Of all reviews predicted as positive, 78% were correctly identified as positive. This shows that the model is more reliable in predicting positive sentiment compared to negative.</br>
#     b. Recall (0.73): Of all actual positive reviews, 73% were correctly identified. This suggests some missed positive reviews (false negatives).</br>
#     c. F1-Score (0.76): Indicates a slightly lower balance between precision and recall compared to the negative class.</br>
# 5. The Multinomial Naive Bayes model performs well in identifying both positive and negative sentiments, with slightly better precision for positive sentiment but higher recall for negative sentiment.

# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

# Step 1: Feature Extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Step 2: Build Naive Bayes Classifier
naive_bayes = MultinomialNB()

# Step 3: Hyperparameter Tuning using GridSearchCV
# Define the hyperparameter grid
param_grid = {'alpha': [0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]}  # 'alpha' is the smoothing parameter

# Use GridSearchCV to find the best 'alpha'
grid_search = GridSearchCV(naive_bayes, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_tfidf, y_train)

# Get the best model
best_naive_bayes = grid_search.best_estimator_

# Step 4: Evaluate the performance on the test set
y_pred = best_naive_bayes.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

print("Best Hyperparameters:", grid_search.best_params_)
print("Accuracy on Test Set:", accuracy)
print("Classification Report:\n", classification_report_result)


# ### Observation for TF-IDF for Feature Extraction, with MultinomialNB
# 1. The optimal value of alpha was determined to be 20.
# 2. The model achieved an overall accuracy of 79%.
# 3. Class 0 (Negative Sentiment): </br>
#     a. Precision (0.75): The model correctly identified 75% of the reviews it predicted as negative. It still has some false positives when labeling negative reviews. </br>
#     b. Recall (0.84): The model successfully identified 84% of all actual negative reviews. This is high, indicating that most negative reviews were detected. </br>
#     c. F1-Score (0.79): Balances precision and recall, showing good overall performance for the negative class.
# 4. Class 1 (Positive Sentiment): </br>
#     a. Precision (0.84): The model correctly identified 84% of the reviews it predicted as positive, which is good precision. </br>
#     b. Recall (0.74): The model successfully identified 74%% of all actual positive reviews, meaning it is better at capturing positive sentiment compared to negative. </br>
#     c. F1-Score (0.79): Shows a reasonably good balance between precision and recall for positive sentiment.
# 5. The TF-IDF-based Multinomial Naive Bayes model achieves a balanced performance, similar to the Bag-of-Words approach. The model exhibits strong precision for positive sentiment and high recall for negative sentiment.

# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

# Step 1: Feature Extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train).toarray()  # Convert to array for GaussianNB
X_test_tfidf = tfidf_vectorizer.transform(X_test).toarray()

# Step 2: Build Gaussian Naive Bayes Classifier
gaussian_nb = GaussianNB()

# Step 3: Hyperparameter Tuning using GridSearchCV
# GaussianNB has limited hyperparameters, mainly 'var_smoothing'
param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]}

# Use GridSearchCV to find the best 'var_smoothing'
grid_search = GridSearchCV(gaussian_nb, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_tfidf, y_train)

# Get the best model
best_gaussian_nb = grid_search.best_estimator_

# Step 4: Evaluate the performance on the test set
y_pred = best_gaussian_nb.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

print("Best Hyperparameters:", grid_search.best_params_)
print("Accuracy on Test Set:", accuracy)
print("Classification Report:\n", classification_report_result)


# ### Observation for TF-IDF for Feature Extraction, with GaussianNB
# 1. The optimal var_smoothing parameter was found to be 0.01.
# 2. The model achieved an overall accuracy of 71%, which is lower compared to the Multinomial Naive Bayes model.
# 3. Class 0 (Negative Sentiment): </br>
#     a. Precision (0.73): Of all reviews predicted as negative, 73% were correctly classified. This reflects a moderate rate of false positives when predicting negative sentiment. </br>
#     b. Recall (0.62): The model successfully identified 62% of actual negative reviews, indicating a significant number of false negatives. </br>
#     c. F1-Score (0.67): The harmonic mean of precision and recall is lower for this class, suggesting room for improvement in correctly classifying negative sentiment.
# 4. Class 1 (Positive Sentiment): </br>
#     a. Precision (0.69): The model correctly identified 69% of the reviews predicted as positive, indicating a relatively high rate of false positives. </br>
#     b. Recall (0.79): The model identified 79% of actual positive reviews, showing better performance in detecting positive sentiment compared to negative sentiment. </br>
#     c. F1-Score (0.74): Indicates a better balance between precision and recall for the positive class.
# 5. The Gaussian Naive Bayes model, when used with TF-IDF feature extraction, shows a lower performance compared to the Multinomial Naive Bayes model. It performs better in detecting positive sentiment (higher recall) but struggles with identifying negative sentiment accurately. This suggests that Gaussian Naive Bayes, which assumes continuous features, might not be well-suited for discrete text data represented by TF-IDF vectors. Further optimization or a different model might yield better results.
