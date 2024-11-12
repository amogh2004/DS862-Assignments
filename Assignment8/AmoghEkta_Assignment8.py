#!/usr/bin/env python
# coding: utf-8

# # DS 862 - ASSIGNMENT 8
# ## AMOGH RANGANATHAIAH (aranganathaiah@sfsu.edu)
# ## EKTA SINGH (esingh@sfsu.edu)
# 
# For this assignment, we will use the ABC News data set found [here](https://www.kaggle.com/therohk/million-headlines). The goal is to create topics from the news.

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import sklearn
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF


# In[2]:


data = pd.read_csv('abcnews-date-text.csv', on_bad_lines='skip')
# We only need the Headlines_text column from the data
# Also this is a huge dataset, let's only use the first 30000 lines
data_text = data['headline_text'][:30000]


# In[3]:


data_text


# In[4]:


# Define two bag-of-words settings for LDA
# Setting 1: Max features = 5000, min_df = 5
vectorizer1 = CountVectorizer(max_features=5000, min_df=5, max_df = 0.95, stop_words='english')
data_matrix1 = vectorizer1.fit_transform(data_text)

# Setting 2: Max features = 10000, min_df = 10
vectorizer2 = CountVectorizer(max_features=10000, min_df=10, max_df = 0.9, stop_words='english')
data_matrix2 = vectorizer2.fit_transform(data_text)

# Function to display top words for each topic
def display_topics(model, feature_names, n_top_words=10):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append(f"Topic #{topic_idx + 1}: " + " ".join(top_words))
    return topics

# Train LDA for both settings with an assumed number of topics
n_topics = 5


# In[5]:


# LDA Model for Setting 1
lda1 = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method = 'batch', learning_decay = 0.7, max_iter = 500)
lda1.fit(data_matrix1)
print("LDA Topics for Setting 1:")
topics1 = display_topics(lda1, vectorizer1.get_feature_names_out())
for topic in topics1:
    print(topic)


# In[6]:


# LDA Model for Setting 2
lda2 = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method = 'batch', learning_decay = 0.8, max_iter = 1000)
lda2.fit(data_matrix2)
print("LDA Topics for Setting 2:")
topics2 = display_topics(lda2, vectorizer2.get_feature_names_out())
for topic in topics2:
    print(topic)


# **LDA Topics for Setting 1**
# 1. Topic 1: Water and Environmental Initiatives
# Words like water, council, boost, funds, budget, and coast suggest a focus on environmental and industry-related topics, possibly around water resource management, coastal funding, and industry support.
# 
# 2. Topic 2: Government and Law Enforcement
# Terms like govt, man, court, police, nsw, and plan indicate topics related to government actions and police involvement, including law enforcement initiatives or court proceedings in New South Wales (NSW) and other regions.
# 
# 3. Topic 3: Iraq War and Conflict
# Words such as iraq, war, police, iraqi, and anti signify topics related to the Iraq war, including discussions on police actions, anti-war sentiments, and hospital or emergency responses.
# 
# 4. Topic 4: Sports and Accidents
# Terms like win, cup, world, final, car, and crash suggest topics covering sports events (like world cups or finals) and perhaps accidents or crashes related to transportation or public events.
# 
# 5. Topic 5: Health and International Relations
# Words like health, sars, talks, drought, korea, north, and aid indicate topics surrounding public health issues (such as SARS), international relations (North Korea), and aid discussions.
# 
# 
# **LDA Topics for Setting 2**
# 1. Topic 1: Government Plans and Local Issues
# With words like plan, concerns, mp, urged, coast, and council, this topic seems to cover government planning and local concerns, with potential focus areas like infrastructure projects or community initiatives.
# 
# 2. Topic 2: Crime and Law Enforcement
# Words like police, man, court, hospital, charged, and trial imply a focus on crime, law enforcement, and legal proceedings, including cases of serious crimes and hospital involvement.
# 
# 3. Topic 3: Health and Environmental Changes
# Terms like sars, water, home, rain, and rise suggest themes around health (e.g., SARS), environmental issues (like rainfall and water levels), and their effects on homes or communities in Australia.
# 
# 4. Topic 4: Iraq War and Casualties
# Words such as iraq, war, iraqi, killed, crash, and baghdad strongly indicate a topic focused on the Iraq war, including casualties, police actions, and military movements.
# 
# 5. Topic 5: Government, Sports, and Local Initiatives
# With terms like govt, claims, cup, world, council, and nsw, this topic appears to combine government statements, support for local events (possibly sports-related), and council involvement in regional activities.
# 
# **Summary of Differences** </br>
# While both settings reveal similar themes (e.g., government actions, health concerns, Iraq war), the focus within each topic slightly varies. Setting 1 includes a more distinct separation between health/international aid and government/law enforcement, while Setting 2 appears to combine some government and community themes, especially around local issues and public events. This highlights how adjusting the CountVectorizer parameters can affect topic composition and specificity.

# In[7]:


from wordcloud import WordCloud

# Visualizing topics with word clouds for the second setting
def plot_word_clouds(model, feature_names, n_top_words=30):
    for topic_idx, topic in enumerate(model.components_):
        top_words = {feature_names[i]: topic[i] for i in topic.argsort()[:-n_top_words - 1:-1]}
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top_words)
        plt.figure()
        plt.title(f"Word Cloud for Topic #{topic_idx + 1}")
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

# Generate word clouds for the second setting topics
plot_word_clouds(lda2, vectorizer2.get_feature_names_out())


# ## NMF

# In[8]:


# Define two TF-IDF settings for NMF
# Setting 1: Max features = 5000, min_df = 5
tfidf_vectorizer1 = TfidfVectorizer(max_features=5000, min_df=5, stop_words='english')
tfidf_matrix1 = tfidf_vectorizer1.fit_transform(data_text)

# Setting 2: Max features = 10000, min_df = 10
tfidf_vectorizer2 = TfidfVectorizer(max_features=10000, min_df=10, stop_words='english')
tfidf_matrix2 = tfidf_vectorizer2.fit_transform(data_text)


# In[9]:


# Function to display top words for each topic
def display_topics(model, feature_names, n_top_words=10):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append(f"Topic #{topic_idx + 1}: " + " ".join(top_words))
    return topics


# In[10]:


# Define a function to find the optimal number of topics based on the reconstruction error
def find_optimal_topics(data_matrix, vectorizer, max_k=10):
    errors = []
    for k in range(1, max_k + 1):
        nmf_model = NMF(n_components=k, random_state=42)
        nmf_model.fit(data_matrix)
        error = nmf_model.reconstruction_err_
        errors.append(error)
    return errors


# In[11]:


# Determine optimal number of topics for both settings
errors1 = find_optimal_topics(tfidf_matrix1, tfidf_vectorizer1)
errors2 = find_optimal_topics(tfidf_matrix2, tfidf_vectorizer2)


# In[12]:


# Plotting reconstruction errors to determine the optimal k
plt.plot(range(1, 11), errors1, label='Setting 1')
plt.plot(range(1, 11), errors2, label='Setting 2')
plt.xlabel('Number of Topics')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.show()


# In[13]:


# Choosing the number of topics with the lowest error for each setting
optimal_k1 = errors1.index(min(errors1)) + 1
optimal_k2 = errors2.index(min(errors2)) + 1


# In[14]:


# NMF Model for Setting 1
nmf1 = NMF(n_components=optimal_k1, max_iter = 500, init = 'nndsvd', random_state=42)
nmf1.fit(tfidf_matrix1)
print(f"NMF Topics for Setting 1 with k={optimal_k1}:")
topics1 = display_topics(nmf1, tfidf_vectorizer1.get_feature_names_out())
for topic in topics1:
    print(topic)


# In[15]:


# NMF Model for Setting 2
nmf2 = NMF(n_components=optimal_k2, max_iter = 1000, init = 'nndsvdar', random_state=42)
nmf2.fit(tfidf_matrix2)
print(f"NMF Topics for Setting 2 with k={optimal_k2}:")
topics2 = display_topics(nmf2, tfidf_vectorizer2.get_feature_names_out())
for topic in topics2:
    print(topic)


# **Summary of Differences** </br>
# Both NMF settings yield very similar themes but with slight differences in focus. Setting 2 generally brings more specificity in court-related topics and combines certain themes, like SARS and sports. The topics demonstrate NMF’s ability to provide a focused interpretation of events within distinct themes like police investigations, governmental actions, war, health crises, and sports, with some variations in how certain themes overlap.

# In[16]:


# Visualizing topics with word clouds for the second setting
def plot_word_clouds(model, feature_names, n_top_words=30):
    for topic_idx, topic in enumerate(model.components_):
        top_words = {feature_names[i]: topic[i] for i in topic.argsort()[:-n_top_words - 1:-1]}
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top_words)
        plt.figure()
        plt.title(f"Word Cloud for Topic #{topic_idx + 1}")
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

# Generate word clouds for topics in second NMF setting
plot_word_clouds(nmf2, tfidf_vectorizer2.get_feature_names_out())


# In[ ]:




