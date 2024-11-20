#!/usr/bin/env python
# coding: utf-8

# # DS 862 - ASSIGNMENT 9
# ## AMOGH RANGANATHAIAH (aranganathaiah@sfsu.edu)
# ## EKTA SINGH (esingh@sfsu.edu)
# 
# For this assignment, we will use the Goodbooks 10k data set found [here](https://github.com/zygmuntz/goodbooks-10k). 

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


pd.set_option('display.max_columns', 500)

books = pd.read_csv('books.csv') # Book metadata
books


# In[3]:


ratings = pd.read_csv('ratings.csv') # User ratings
ratings


# In[4]:


len(ratings['user_id'].unique())


# ## Data Preprocessing

# In[5]:


# Merge the two datasets
merged_data = pd.merge(books, ratings, on='book_id')[['user_id', 'book_id', 'rating', 'original_title']]


# In[6]:


merged_data = merged_data[merged_data.user_id <= 10000]


# In[7]:


merged_data


# ## User-Based Collaborative Filtering

# In[8]:


# Create a user-item interaction matrix
ratings_matrix = merged_data.pivot_table(index='user_id', columns='book_id', values='rating')

# Replace missing values with 0
ratings_matrix.fillna(0, inplace=True)

ratings_matrix


# In[9]:


from scipy.sparse import csr_matrix
from scipy.spatial.distance import euclidean

# Convert ratings matrix to a sparse matrix
ratings_sparse = csr_matrix(ratings_matrix.values)

def calculate_similarity_sparse(user_id, data_sparse, top_n):
    """
    Calculate similarities for a user with all others in a sparse matrix.
    """
    user_vector = data_sparse[user_id - 1].toarray().flatten()  # Flatten to ensure 1D
    similarities = []
    for other_user in range(data_sparse.shape[0]):
        if other_user == user_id - 1:  # Skip self-comparison
            continue
        other_vector = data_sparse[other_user].toarray().flatten()  # Flatten to ensure 1D
        # Compute Euclidean distance and similarity
        dist = euclidean(user_vector, other_vector)
        similarity = 1 / (1 + dist)  # Normalize to range [0, 1]
        similarities.append((other_user + 1, similarity))  # Add 1 to match user IDs
    # Sort and get top N
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
    return similarities

def user_based_cf_sparse(user_id, n_neighbors, top_n, data_sparse, original_data):
    # Get top N neighbors
    neighbors = calculate_similarity_sparse(user_id, data_sparse, n_neighbors)
    neighbors = dict(neighbors)
    
    # Predict ratings for unseen books
    unseen_books = np.where(data_sparse[user_id - 1].toarray().flatten() == 0)[0]  # Unrated books for the user
    predicted_ratings = {}
    for book_id in unseen_books:
        # Aggregate neighbor ratings weighted by similarity
        numerator, denominator = 0, 0
        for neighbor_id, similarity in neighbors.items():
            rating = data_sparse[neighbor_id - 1, book_id]  # Neighbor's rating for the book
            if rating > 0:
                numerator += similarity * rating
                denominator += similarity
        if denominator > 0:
            predicted_ratings[book_id] = numerator / denominator

    # Sort predictions and fetch top N
    predicted_ratings = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
    recommendations = [
        (
            book_id + 1,  # Adjust book ID for zero-based indexing
            original_data.loc[original_data['book_id'] == book_id + 1, 'original_title'].values[0],
            rating,
        )
        for book_id, rating in predicted_ratings
    ]
    return recommendations

# Get top 15 recommendations for user 1839
ubcf_recommendations_sparse = user_based_cf_sparse(1839, 100, 15, ratings_sparse, books)

# Display recommendations
for rank, (book_id, title, predicted_rating) in enumerate(ubcf_recommendations_sparse, start=1):
    print(f"Rank {rank}: {title} (Predicted Rating: {predicted_rating:.2f})")


# ## Item-Based Collaborative Filtering

# In[10]:


from sklearn.metrics.pairwise import cosine_similarity

# Calculate the item-item similarity matrix using cosine similarity
item_sim = cosine_similarity(ratings_sparse.T)  # Transpose for item-based similarity
item_sim_df = pd.DataFrame(item_sim, index=ratings_matrix.columns, columns=ratings_matrix.columns)

def item_based_cf(user_id, n_neighbors, top_n, similarity_matrix, data_sparse, original_data):
    """
    Generate top N recommendations for a user using Item-Based Collaborative Filtering.
    """
    # Get the user's ratings
    user_ratings = data_sparse[user_id - 1].toarray().flatten()
    
    # Predict ratings for unseen books
    unseen_books = np.where(user_ratings == 0)[0]  # Unrated books for the user
    predicted_ratings = {}
    for book_id in unseen_books:
        # Find similar items to the current book
        similar_items = similarity_matrix[book_id]
        similar_items = pd.Series(similar_items).sort_values(ascending=False)[1:n_neighbors+1]
        
        # Aggregate the ratings of similar items weighted by similarity
        numerator, denominator = 0, 0
        for sim_item_id, similarity in similar_items.items():
            rating = user_ratings[sim_item_id]
            if rating > 0:
                numerator += similarity * rating
                denominator += similarity
        if denominator > 0:
            predicted_ratings[book_id] = numerator / denominator

    # Sort predictions and fetch top N
    predicted_ratings = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
    recommendations = [
        (
            book_id + 1,  # Adjust book ID for zero-based indexing
            original_data.loc[original_data['book_id'] == book_id + 1, 'original_title'].values[0],
            rating,
        )
        for book_id, rating in predicted_ratings
    ]
    return recommendations

# Get top 15 recommendations for user 1839
ibcf_recommendations = item_based_cf(1839, 100, 15, item_sim, ratings_sparse, books)

# Display recommendations
for rank, (book_id, title, predicted_rating) in enumerate(ibcf_recommendations, start=1):
    print(f"Rank {rank}: {title} (Predicted Rating: {predicted_rating:.2f})")

# Store the recommendations for later comparison
ibcf_recommendations_sparse = ibcf_recommendations


# ## Matrix Factorization

# In[11]:


# Function for Matrix Factorization
def matrix_factorization(R, P, Q, K, steps=5, alpha=0.001, beta=0.01):
    """
    Inputs:
    R     : The ratings (M x N matrix)
    P     : User-feature matrix (M x K)
    Q     : Item-feature matrix (K x N)
    K     : Number of latent features
    steps : Number of iterations
    alpha : Learning rate
    beta  : Regularization parameter
    Outputs:
    Final matrices P and Q
    """
    for step in range(steps):
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if R[i][j] > 0:  # Only consider existing ratings
                    eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                    for k in range(K):
                        P[i][k] += alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] += alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        # Calculate total error
        eR = np.dot(P, Q)
        e = 0
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if R[i][j] > 0:
                    e += (R[i][j] - np.dot(P[i, :], Q[:, j])) ** 2
                    for k in range(K):
                        e += (beta / 2) * (P[i][k] ** 2 + Q[k][j] ** 2)
        if e < 0.001:  # Convergence threshold
            break
    return P, Q

# Initialize matrices
np.random.seed(862)
num_users, num_items = ratings_sparse.shape
K = 3  # Latent factors
P = np.random.rand(num_users, K)  # User-feature matrix
Q = np.random.rand(K, num_items)  # Item-feature matrix
R = ratings_sparse.toarray()  # Convert sparse matrix to dense

# Fit the model
P, Q = matrix_factorization(R, P, Q, K, steps=5, alpha=0.001, beta=0.01)

# Predict ratings
predicted_ratings = np.dot(P, Q)
predicted_ratings_df = pd.DataFrame(predicted_ratings, index=ratings_matrix.index, columns=ratings_matrix.columns)

# Function to get recommendations
def get_mf_recommendations(user_id, top_n, predicted_ratings_matrix, original_data):
    """
    Get top N recommendations for a user using the predicted ratings matrix.
    """
    user_predictions = predicted_ratings_matrix.loc[user_id]
    # Filter out already rated items
    unseen_books = ratings_matrix.loc[user_id][ratings_matrix.loc[user_id] == 0].index
    user_predictions = user_predictions[unseen_books]
    # Sort predictions by rating
    top_recommendations = user_predictions.sort_values(ascending=False).head(top_n)
    recommendations = [
        (
            book_id,
            original_data.loc[original_data['book_id'] == book_id, 'original_title'].values[0],
            rating,
        )
        for book_id, rating in top_recommendations.items()
    ]
    return recommendations

# Get top 15 recommendations for user 1839
mf_recommendations = get_mf_recommendations(1839, 15, predicted_ratings_df, books)

# Display recommendations
for rank, (book_id, title, predicted_rating) in enumerate(mf_recommendations, start=1):
    print(f"Rank {rank}: {title} (Predicted Rating: {predicted_rating:.2f})")

# Store the recommendations for later comparison
mf_recommendations_stored = mf_recommendations


# ## SVD++

# In[12]:


get_ipython().system('pip install scikit-surprise')


# In[13]:


from surprise import Reader
from surprise import Dataset
from surprise.prediction_algorithms.matrix_factorization import SVD


# In[14]:


# Set up the reader class
reader = Reader(rating_scale=(1,5))


# In[15]:


# Load the dataframe. Use the merged data from above (not the pivoted data)
data = Dataset.load_from_df(merged_data[['user_id', 'book_id', 'rating']], reader)


# In[16]:


# Build the train set
svd_data = data.build_full_trainset()


# In[17]:


# Instantiate the SVD model
svd_model = SVD(
    n_factors=5,  # Number of latent factors
    lr_all=0.01,  # Learning rate for all parameters
    reg_all=0.1,  # Regularization for all parameters
    random_state=862  # For reproducibility
)

# Fit the SVD model on the training data
svd_model.fit(svd_data)

# Predict ratings for user 1839
def svd_recommendations(user_id, top_n, svd_model, original_data):
    """
    Generate top N recommendations for a user using SVD model predictions.
    """
    # Get a list of all books
    all_books = merged_data['book_id'].unique()
    # Get books already rated by the user
    rated_books = merged_data[merged_data['user_id'] == user_id]['book_id'].unique()
    # Filter out books already rated
    unseen_books = [book for book in all_books if book not in rated_books]
    
    # Predict ratings for unseen books
    predictions = [
        (book_id, svd_model.predict(user_id, book_id).est) for book_id in unseen_books
    ]
    # Sort predictions by rating
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
    
    # Fetch book titles for recommendations
    recommendations = [
        (
            book_id,
            original_data.loc[original_data['book_id'] == book_id, 'original_title'].values[0],
            rating,
        )
        for book_id, rating in predictions
    ]
    return recommendations

# Get top 15 recommendations for user 1839
svd_recommendations_user_1839 = svd_recommendations(1839, 15, svd_model, books)

# Display recommendations
for rank, (book_id, title, predicted_rating) in enumerate(svd_recommendations_user_1839, start=1):
    print(f"Rank {rank}: {title} (Predicted Rating: {predicted_rating:.2f})")

# Store the recommendations for later comparison
svd_recommendations_stored = svd_recommendations_user_1839


# In[18]:


# Prepare the recommendations into a structured format
recommendations_df = pd.DataFrame({
    "User-Based CF": [title for _, title, _ in ubcf_recommendations_sparse],
    "Item-Based CF": [title for _, title, _ in ibcf_recommendations_sparse],
    "Matrix Factorization": [title for _, title, _ in mf_recommendations_stored],
    "SVD": [title for _, title, _ in svd_recommendations_stored],
})

# Display the DataFrame
recommendations_df

