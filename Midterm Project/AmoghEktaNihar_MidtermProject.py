#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install prince


# In[2]:


# !pip install gower


# In[3]:


# !pip install scikit-learn-extra


# In[4]:


import warnings
warnings.filterwarnings('ignore')


# ## Midterm Project
# 
# ### Amogh Ranganathaiah (aranganathaiah@sfsu.edu)
# ### Ekta Singh (esingh@sfsu.edu)
# ### Nihar Shah (nshah4@sfsu.edu)

# ### PART 1: Dimension reduction

# In[5]:


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import prince
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import gower
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import normalized_mutual_info_score


# In[6]:


# Load data set
data = pd.read_csv('train.csv')
data = data.drop('Id', axis=1)

# Remove columns with too many missing values
data = data.drop(data.columns[data.isnull().sum() > 30], axis=1)

# Remove remaining missing values
data.dropna(inplace=True)

# Separate categorical and numerical features
# Categorical features are text-based and identified using 'object' dtype
categorical_features = data.select_dtypes(include=['object'])
numerical_features = data.select_dtypes(exclude=['object'])

# Split categorical and numerical features into training and test sets
# 20% of the data is reserved for testing, with the remaining 80% used for training
# random_state ensures reproducibility of results
X_train_cat, X_test_cat, X_train_num, X_test_num = train_test_split(
    categorical_features, numerical_features, test_size=0.2, random_state=42)


# 1. Dropping columns: Columns with excessive missing values are removed since they may negatively impact the model’s performance.
# 2. Handling missing data: Any rows with remaining missing values are removed to ensure the data is complete.
# 3. Categorical and numerical separation: Categorical features (text-based) and numerical features (numbers) are separated, as they may need different preprocessing steps.

# In[7]:


# Ensure train and test categorical data have the same levels
keep = X_train_cat.nunique() == X_test_cat.nunique()
X_train_cat = X_train_cat[X_train_cat.columns[keep]]
X_test_cat = X_test_cat[X_test_cat.columns[keep]]

# Ensure the classes in training and testing are the same
keep = []
for i in range(X_train_cat.shape[1]):
    keep.append(all(np.sort(X_train_cat.iloc[:, i].unique()) == np.sort(X_test_cat.iloc[:, i].unique())))
X_train_cat = X_train_cat[X_train_cat.columns[keep]]
X_test_cat = X_test_cat[X_test_cat.columns[keep]]


# In[8]:


# Dimension Reduction

# Scaling the numerical features
# StandardScaler standardizes the data by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

# Apply PCA (Principal Component Analysis) on scaled numerical features
pca = PCA(n_components=35) 
X_train_pca = pca.fit_transform(X_train_num_scaled)
X_test_pca = pca.transform(X_test_num_scaled)

# Apply MCA (Multiple Correspondence Analysis) on categorical features
mca = prince.MCA(n_components=1) 
X_train_mca = mca.fit_transform(X_train_cat)
X_test_mca = mca.transform(X_test_cat)


# 1. PCA reduces the dimensionality of the numerical data, keeping only the most important components
# 2. n_components=35: This retains 35 principal components based on prior hyperparameter tuning
# 3. Error was encountered beyond 36 components, due to the number of features being less than 36
# 4. MCA is similar to PCA but is used for categorical data to reduce dimensionality
# 5. n_components=1: Reduces the categorical data to a single dimension

# In[9]:


# Combine PCA and MCA results for training and test sets
X_train_combined = np.hstack((X_train_pca, X_train_mca))
X_test_combined = np.hstack((X_test_pca, X_test_mca))

# Perform Ridge regression on the reduced features
y_train = data.loc[X_train_cat.index, 'SalePrice']
y_test = data.loc[X_test_cat.index, 'SalePrice']

ridge = Ridge(alpha=0.5)
ridge.fit(X_train_combined, y_train)
y_pred_test = ridge.predict(X_test_combined)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred_test)
print(f'Ridge Regression MSE (on reduced features): {mse:.2f}')
print(f'Ridge Regression RMSE (on reduced features): {mse**0.5:.2f}')


# ### Interpretation of the Results:
# 1. MSE (3630.04) shows the overall squared error magnitude, but it's harder to interpret directly.
# 2. RMSE (60.25) provides a clearer idea of how far off your model's predictions are from the actual values. A lower RMSE is preferred, as it indicates better predictive performance.
# 
# During the process of model tuning, we opted not to change the alpha value significantly because increasing or decreasing it too much could result in overfitting or underfitting the model. In Ridge Regression, the alpha parameter controls the amount of regularization applied to the model—higher values of alpha increase regularization, while lower values reduce it.
# 
# Through experimentation, we found a suitable alpha value that balances the model's bias and variance. Increasing the alpha beyond this point would overly constrain the model and risk underfitting, while decreasing it too much would remove the regularization effect, leading to overfitting, especially after dimensionality reduction. Therefore, we settled on the current alpha value as it produced the best balance between model complexity and prediction accuracy, as reflected in the MSE and RMSE.

# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns

# Calculate residuals for both training and test sets
y_pred_train = ridge.predict(X_train_combined)
residuals_train = y_train - y_pred_train
residuals_test = y_test - y_pred_test

# Residual plot for the training set
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_pred_train, residuals_train, alpha=0.5, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals vs Fitted Values (Train Set)')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')

# Residual plot for the test set
plt.subplot(1, 2, 2)
plt.scatter(y_pred_test, residuals_test, alpha=0.5, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals vs Fitted Values (Test Set)')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')

# Show plots
plt.tight_layout()
plt.show()

# Residual distribution plot (Test Set)
plt.figure(figsize=(8, 6))
sns.histplot(residuals_test, kde=True, color='purple')
plt.title('Residual Distribution (Test Set)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


# In[11]:


# BASELINE: Regression on Original Data

# Create dummy variables for categorical features
X_train_cat_dummies = pd.get_dummies(X_train_cat, drop_first=True)
X_test_cat_dummies = pd.get_dummies(X_test_cat, drop_first=True)

# Combine numerical and categorical data
X_train_original = pd.concat([pd.DataFrame(X_train_num_scaled, index=X_train_num.index), X_train_cat_dummies], axis=1)
X_test_original = pd.concat([pd.DataFrame(X_test_num_scaled, index=X_test_num.index), X_test_cat_dummies], axis=1)

# Ensure all column names are strings
X_train_original.columns = X_train_original.columns.astype(str)
X_test_original.columns = X_test_original.columns.astype(str)


# In[12]:


# Fit Ridge regression on the original data
ridge.fit(X_train_original, y_train)
y_pred_test_original = ridge.predict(X_test_original)

# Evaluate the baseline model
mse_original = mean_squared_error(y_test, y_pred_test_original)
print(f'Ridge Regression MSE (on original features): {mse_original:.2f}')
print(f'Ridge Regression RMSE (on reduced features): {mse_original**0.5:.2f}')


# ### Interpretation of the Results:
# 1. The higher MSE (6646.60) and RMSE (81.53) on the original features indicate that the model performs worse on the full dataset compared to the reduced feature set.
# 2. The RMSE of 81.53 shows that the model has a larger average prediction error when using all features, which may indicate that the original feature set includes irrelevant or redundant features that add noise rather than improving predictive power.
# 
# The results on the reduced feature set (MSE: 6646.60, RMSE: 81.53) are better, showing that dimensionality reduction (via PCA and MCA) helped improve the model's predictive accuracy. By removing less important or redundant features, the model becomes more efficient and generalizes better, as indicated by the lower error metrics.
# 
# Just as with the reduced feature set, we chose not to significantly adjust the alpha parameter in the Ridge Regression model to avoid overfitting. Increasing the alpha too much on the original data could have overly constrained the model, while decreasing it would have led to overfitting, especially given the presence of many original features.

# In[13]:


# Calculate residuals for both training and test sets on the original data
y_pred_train_original = ridge.predict(X_train_original)
residuals_train_original = y_train - y_pred_train_original
residuals_test_original = y_test - y_pred_test_original

# Residual plot for the training set (Original Data)
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_pred_train_original, residuals_train_original, alpha=0.5, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals vs Fitted Values (Train Set) - Original Data')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')

# Residual plot for the test set (Original Data)
plt.subplot(1, 2, 2)
plt.scatter(y_pred_test_original, residuals_test_original, alpha=0.5, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals vs Fitted Values (Test Set) - Original Data')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')

# Show plots
plt.tight_layout()
plt.show()

# Residual distribution plot for test set (Original Data)
plt.figure(figsize=(8, 6))
sns.histplot(residuals_test_original, kde=True, color='purple')
plt.title('Residual Distribution (Test Set) - Original Data')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


# ### PART 2: Clustering Analysis

# In[14]:


import pandas as pd
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, normalized_mutual_info_score

# Compute Gower distance matrix for the full data (no train/test split)
full_data = pd.concat([numerical_features, categorical_features], axis=1)

# This line computes the Gower distance matrix, which is a measure that handles mixed numerical and categorical data. 
# The resulting matrix is square and contains the pairwise distances between every pair of rows (instances) in your dataset. 
# Each entry in the matrix represents the Gower distance between two data points.
gower_dist = gower.gower_matrix(full_data)

# Bin the response variable (SalePrice) into 3 groups using qcut
binned_price = pd.qcut(data['SalePrice'], q=2, labels=False)

# Ensure binned_price has the same index as full_data (important for alignment)
binned_price = binned_price.loc[full_data.index]

# Define a custom cross-validator
# The K-medoids algorithm requires a square distance matrix when fitting (training) the model. 
# However, during prediction, we want to use the distances between the test samples and the medoids selected during training. 
# Therefore, the matrix provided for test samples is rectangular, reflecting the test-to-train distances.
def custom_cv(X, y, cv=3):
    skf = StratifiedKFold(n_splits=cv)
    for train_idx, test_idx in skf.split(X, y):
        # Since we are using precomputed distance, ensure we select the correct rows and columns
        train_dist = X[np.ix_(train_idx, train_idx)]
        test_dist = X[np.ix_(test_idx, train_idx)]  # Distance between test samples and training medoids
        yield (train_dist, test_dist, y.iloc[train_idx], y.iloc[test_idx])


# Training vs. Test Separation: During cross-validation, you cannot use test samples to determine cluster centroids (medoids). 
# Hence, we only use distances between test samples and the medoids selected from the 
# training data to predict the clusters for the test set.

# When using precomputed distances, we must handle the distance matrix carefully to ensure that the 
# model fits on training data and predicts using the appropriate medoids for the test data.

# Define a custom scorer for Normalized Mutual Information (NMI)
def nmi_scorer(estimator, X_train, y_train, X_test, y_test):
    # Fit the K-medoids model on the training set
    estimator.fit(X_train)
    
    # Predict the cluster labels for the test set based on the training medoids
    labels_test = estimator.predict(X_test)
    
    # Return NMI between the predicted labels and the true labels of the test set
    return normalized_mutual_info_score(y_test, labels_test)

# Create a custom scorer based on NMI
nmi_score_custom = make_scorer(nmi_scorer, greater_is_better=True, needs_proba=False)

# Tuning parameter grid
param_dist = {'n_clusters': np.arange(2, 10)}

# Custom implementation of the cross-validation loop
best_score = -np.inf
best_param = None

# Iterate through the number of clusters
for n_clusters in param_dist['n_clusters']:
    scores = []
    
    # Perform custom cross-validation
    for train_dist, test_dist, y_train, y_test in custom_cv(gower_dist, binned_price, cv=3):
        # Apply KMedoids with precomputed distance
        kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=42)
        
        # Evaluate NMI on the test set
        score = nmi_scorer(kmedoids, train_dist, y_train, test_dist, y_test)
        scores.append(score)

    # Get the average score across all folds
    mean_score = np.mean(scores)
    if mean_score > best_score:
        best_score = mean_score
        best_param = n_clusters

# Print the best number of clusters and corresponding NMI score
print(f'Best number of clusters: {best_param} with NMI Score: {best_score:.2f}')


# The algorithm found that 2 clusters were optimal based on our configuration. This could suggest that the dataset is not highly differentiable into multiple clusters or that the main differentiating features are too subtle for K-medoids to detect beyond 2 broad groupings.

# **Why Do We Need This Approach?**
# 1. Precomputed Distance Matrix: The Gower distance matrix contains pairwise distances between all data points, but we can't simply split this matrix like we would with raw data (e.g., numerical or categorical features). The challenge with cross-validation in this context is to ensure that the training and test distance matrices are correctly aligned to avoid shape mismatch errors.
# 
# 2. Clustering Evaluation: NMI is used to evaluate the agreement between the predicted clusters and the actual labels (binned_price). By carefully using the train-test splits, we ensure that the model is evaluated on unseen data, giving us a fair assessment of clustering performance.

# The reason why the NMI Score is only 0.40 could be because:
# 1. Clustering algorithms, especially when using a mixed distance metric like Gower distance, might struggle with datasets that have many categorical features and missing values. If the data is sparse or the categorical variables dominate the distance calculations, the clusters formed might not align well with the ground truth labels.
# 2. K-medoids clustering tends to form clusters based on the central representative points (medoids), but when you have a significant amount of categorical data, it can be challenging to find clear-cut clusters. This can result in low mutual information when compared to the SalePrice labels, as SalePrice might be more influenced by specific numerical features (like OverallQual, GrLivArea, etc.) than categorical ones.
# 3. The algorithm found that 2 clusters were optimal based on your configuration. This could suggest that the dataset is not highly differentiable into multiple clusters or that the main differentiating features are too subtle for K-medoids to detect beyond 2 broad groupings.
# 4. With 81 features, many of which might be irrelevant or redundant, the distance calculation may suffer from the "curse of dimensionality." This dilutes the strength of the clusters, as noise from irrelevant features may obscure important patterns.

# In[15]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids

# Store the number of clusters and their corresponding NMI scores
cluster_range = param_dist['n_clusters']
nmi_scores = []

# Perform custom cross-validation for each number of clusters
for n_clusters in cluster_range:
    scores = []
    for train_dist, test_dist, y_train, y_test in custom_cv(gower_dist, binned_price, cv=3):
        kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=42)
        score = nmi_scorer(kmedoids, train_dist, y_train, test_dist, y_test)
        scores.append(score)
    mean_score = np.mean(scores)
    nmi_scores.append(mean_score)

# Elbow Plot: NMI Score vs. Number of Clusters
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, nmi_scores, marker='o', color='b')
plt.title('NMI Score vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('NMI Score')
plt.grid(True)
plt.show()

# PCA for visualizing the clustering in 2D space
pca = PCA(n_components=2)
full_data_pca = pca.fit_transform(gower_dist)

# Fit the KMedoids model with the best number of clusters
best_kmedoids = KMedoids(n_clusters=best_param, metric='precomputed', random_state=42)
best_kmedoids.fit(gower_dist)
labels = best_kmedoids.labels_

# Cluster Assignment Plot
plt.figure(figsize=(8, 6))
plt.scatter(full_data_pca[:, 0], full_data_pca[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
plt.title(f'Cluster Assignments with {best_param} Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Label')
plt.grid(True)
plt.show()

