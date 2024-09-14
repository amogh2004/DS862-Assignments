# DS 862 - ASSIGNMENT 1
# AMOGH RANGANATHAIAH (aranganathaiah@sfsu.edu)
# EKTA SINGH (esingh@sfsu.edu)

# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import KernelPCA
from sklearn.manifold import LocallyLinearEmbedding, Isomap
from sklearn.metrics import accuracy_score

# Load and preprocess the dataset
data = pd.read_csv('fashion-mnist_train-1.csv')
data.drop_duplicates(inplace=True)

# Preprocess the dataset (using first 1500 samples for consistency)
X = data.drop('label', axis=1)
y = data['label']
X = X.loc[0:1500, :]
y = y.loc[0:1500]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=862, stratify=y)

# Define classifiers to use
classifiers = {
    'RandomForest': RandomForestClassifier(random_state=862),
    'KNN': KNeighborsClassifier()
}

# Define dimensionality reduction techniques to use
dim_reductions = {
    'KernelPCA': KernelPCA(kernel='rbf', n_components=15, n_jobs=-1),  # Increased n_components to 15
    'LLE': LocallyLinearEmbedding(n_components=15, n_neighbors=10),
    'Isomap': Isomap(n_components=15, n_neighbors=10, n_jobs=-1)
}

# Define the grid search parameters for classifiers
param_grids = {
    'RandomForest': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    'KNN': {
        'classifier__n_neighbors': [3, 5, 7, 9],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__p': [1, 2]  # L1 and L2 norms
    }
}

# Use Stratified K-Fold cross-validation
cv = StratifiedKFold(n_splits=5)


# Function to evaluate model performance with dimensionality reduction
def evaluate_model(dim_reduction, classifier_name, classifier):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Added scaling step
        ('dim_reduction', dim_reduction),
        ('classifier', classifier)
    ])

    param_grid = param_grids[classifier_name]

    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, grid_search.best_params_


# Iterate over the dimension reduction techniques and classifiers
results = {}
for dim_name, dim_reduction in dim_reductions.items():
    for clf_name, clf in classifiers.items():
        accuracy, best_params = evaluate_model(dim_reduction, clf_name, clf)
        results[(dim_name, clf_name)] = (accuracy, best_params)


# Now evaluating without dimensionality reduction
def evaluate_original_data(classifier_name, classifier):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Scaling
        ('classifier', classifier)
    ])

    param_grid = param_grids[classifier_name]

    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, grid_search.best_params_

# What is the best combination according to your accuracy score on the test set?
# 1. The best combination according to the accuracy score is KernelPCA with RandomForest,
# achieving an accuracy of 75.53%.
# 2. KernelPCA performed the best with both classifiers (RandomForest and KNN), particularly with RandomForest,
# suggesting that it was able to extract the most useful features from the data while preserving
# important non-linear relationships.
# 3. Across all three dimensionality reduction techniques, RandomForest consistently outperformed KNN.
# This suggests that the ensemble method of RandomForest was more effective in capturing the complex patterns
# in the data compared to KNN's instance-based approach.
# 4. The accuracy of each classifier was influenced by the choice of dimensionality reduction technique.
# KernelPCA consistently provided better results, likely because of its ability to handle non-linear transformations.

# Evaluate original data (no dimensionality reduction)
for clf_name, clf in classifiers.items():
    accuracy, best_params = evaluate_original_data(clf_name, clf)
    results[('OriginalData', clf_name)] = (accuracy, best_params)

# Display the results
results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy', 'Best Parameters'])
print(results_df)


# Method 2: We tried executing the code again completely for all-in model, and got a better accuracy for this model.
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

print()
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
