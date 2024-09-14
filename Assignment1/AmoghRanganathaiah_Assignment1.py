# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 08:09:44 2024

@author: 918839342
"""

# For this assignment we are going to use the Fashion MNIST data set.

# The fashion mnist data set is composed of 60,000 small square 28x28 grayscale images of 10 types of clothing items:
# such as shoes, t-shirts, dress. Each item label is mapped to a 0-9 integer.

#     0: T-shirt/top
#     1: Trouser
#     2: Pullover
#     3: Dress
#     4: Coat
#     5: Sandal
#     6: Shirt
#     7: Sneaker
#     8: Bag
#     9: Ankle boot

# Your job is to apply the dimension reduction technique we learned, and combined with the classification methods you
# learned from DS861, to build a classifier. I will load the data for you
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('fashion-mnist_train-1.csv')
data.drop_duplicates(inplace=True)
X = data.drop('label', axis=1)
y = data.label

# To make your life easier, let's use only the first 1500 data points.
X = X.loc[0:1500, ]
y = y.loc[0:1500, ]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=862)

# Here is where you import all of your libraries
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.decomposition import KernelPCA

# Here you will practice dimension reduction techniques learned in class. In particular, you will have a chance to
# preprocess the data with one of the dimension reduction techniques learned, then paired with a classification model
# to evaluate the overall performance. Recall that we have learned the following classifiers in DS861: Logistic
# Regression, Decision Tree, Random Forest, Boosting, KNN. This data set is a multi-level data set, hence you should
# think about which model is appropriate (or not appropriate).

# Some general rules you should follow:

#     Tune your dimension reduction technique
#     Tune your model
#     Select your hyperparameters based on a hold-out set (either via CV or train/validate/test split)
#     Report the accuracy on the test set

# You will be using kernel PCA, LLE, and Isomap. For each dimension reduction technique, perform classification with
# two classifiers (you can use the same classifiers across the different dimension reduction techniques, but you will
# still need to specifically tune each time.


# Kernel PCA
kpca = KernelPCA(n_components=2, kernel="rbf", random_state=862, n_jobs=-1)
X_kpca_train = kpca.fit_transform(X_train)
X_kpca_test = kpca.transform(X_test)

# Classifiers
classifiers = {
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier()
}

# Hyperparameters for tuning
param_grids = {
    "RandomForest": {
        "n_estimators": [50, 100],
        "max_depth": [10, 20, 30],
        "random_state": [862]
    },
    "KNN": {
        "n_neighbors": [3, 5, 7],
        "weights": ['uniform', 'distance']
    }
}

# Perform classification using both Random Forest and KNN
for name, clf in classifiers.items():
    grid_search = GridSearchCV(clf, param_grids[name], cv=5)
    grid_search.fit(X_kpca_train, y_train)
    y_pred = grid_search.predict(X_kpca_test)

    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Accuracy for {name} with Kernel PCA: {accuracy_score(y_test, y_pred)}")
print()
print()


# LLE
# LLE + Random Forest and KNN
from sklearn.manifold import LocallyLinearEmbedding

# Apply LLE for dimensionality reduction
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=862)
X_lle_train = lle.fit_transform(X_train)
X_lle_test = lle.transform(X_test)

# Classifiers
classifiers = {
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier()
}

# Hyperparameters for tuning
param_grids = {
    "RandomForest": {
        "n_estimators": [50, 100],
        "max_depth": [10, 20, 30],
        "random_state": [862]
    },
    "KNN": {
        "n_neighbors": [3, 5, 7],
        "weights": ['uniform', 'distance']
    }
}

# Perform classification using both Random Forest and KNN
for name, clf in classifiers.items():
    grid_search = GridSearchCV(clf, param_grids[name], cv=5)
    grid_search.fit(X_lle_train, y_train)
    y_pred = grid_search.predict(X_lle_test)

    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Accuracy for {name} with LLE: {accuracy_score(y_test, y_pred)}")
print()
print()


# Isomap
# Isomap + Random Forest and KNN
from sklearn.manifold import Isomap

# Apply Isomap for dimensionality reduction
isomap = Isomap(n_components=2, n_neighbors=10, n_jobs=-1)
X_isomap_train = isomap.fit_transform(X_train)
X_isomap_test = isomap.transform(X_test)

# Classifiers
classifiers = {
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier()
}

# Hyperparameters for tuning
param_grids = {
    "RandomForest": {
        "n_estimators": [50, 100],
        "max_depth": [10, 20, 30],
        "random_state": [862]
    },
    "KNN": {
        "n_neighbors": [3, 5, 7],
        "weights": ['uniform', 'distance']
    }
}

# Perform classification using both Random Forest and KNN
for name, clf in classifiers.items():
    grid_search = GridSearchCV(clf, param_grids[name], cv=5)
    grid_search.fit(X_isomap_train, y_train)
    y_pred = grid_search.predict(X_isomap_test)

    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Accuracy for {name} with Isomap: {accuracy_score(y_test, y_pred)}")
print()
print()


# What is the best combination according to your accuracy score on the test set?
# 1. The best combination according to the accuracy score is RandomForest with LLE, with an accuracy of 66.49%.
# 2. KNN with LLE is the next best combination, achieving 64.89% accuracy.
# 3. LLE appears to be the most effective dimensionality reduction technique
# for this dataset, as it preserves more information, leading to higher classification accuracy.
# 4. RandomForest consistently performs better than KNN across all dimensionality reduction techniques.


# Now using the original data set (i.e. not reduced data) and the two classifiers you chose,
# run the procedure again, but this time without any dimension reduction. Make sure you tune
# your classifiers. Which result is better? Using the original data set or the reduced data set?

data = pd.read_csv('fashion-mnist_train-1.csv')
data.drop_duplicates(inplace=True)
X = data.drop('label', axis=1)
y = data.label

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=862)

# Classifiers
classifiers = {
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier()
}

# Hyperparameters for tuning
param_grids = {
    "RandomForest": {
        "n_estimators": [50, 100],
        "max_depth": [10, 20, 30],
        "random_state": [862]
    },
    "KNN": {
        "n_neighbors": [3, 5, 7],
        "weights": ['uniform', 'distance']
    }
}

# Perform classification using both Random Forest and KNN
for name, clf in classifiers.items():
    grid_search = GridSearchCV(clf, param_grids[name], cv=5)
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)

    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Accuracy for {name} with original data and without dimensionality reduction: {accuracy_score(y_test, y_pred)}")

# What's your observation?
# 1. Using the original dataset without any dimensionality reduction yields significantly better results than using the
# dimensionality reduction techniques (Kernel PCA, LLE, and Isomap) for both RandomForest and KNN.
# 2. This suggests that the dimensionality reduction techniques may be losing important information necessary for
# accurate classification.
# 3. The best overall result is RandomForest without dimensionality reduction, achieving an accuracy of 87.71%.
