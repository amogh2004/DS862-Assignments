# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import KernelPCA
from sklearn.manifold import LocallyLinearEmbedding, Isomap
from sklearn.metrics import accuracy_score

# Load and preprocess the dataset
data = pd.read_csv('fashion-mnist_train-1.csv')
data.drop_duplicates(inplace=True)

X = data.drop('label', axis=1)
y = data['label']

# Only use the first 1500 data points
X = X.loc[0:1500, :]
y = y.loc[0:1500]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=862)

# Define classifiers to use
classifiers = {
    'RandomForest': RandomForestClassifier(random_state=862),
    'KNN': KNeighborsClassifier()
}

# Define dimensionality reduction techniques to use
dim_reductions = {
    'KernelPCA': KernelPCA(kernel='rbf', n_jobs=-1),
    'LLE': LocallyLinearEmbedding(),
    'Isomap': Isomap()
}

# Function to evaluate model performance
def evaluate_model(pipeline, param_grid):
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, grid_search.best_params_

# Results storage
results = []

# Loop through dimensionality reduction techniques and classifiers
for dr_name, dr in dim_reductions.items():
    for clf_name, clf in classifiers.items():
        # Create pipeline
        pipeline = Pipeline(steps=[('dim_reduction', dr), ('classifier', clf)])
        
        # Define parameter grids for tuning
        if clf_name == 'KNN':
            param_grid = {
                'dim_reduction__n_components': [10, 20, 30],
                'classifier__n_neighbors': [3, 5, 7]
            }
        elif clf_name == 'RandomForest':
            param_grid = {
                'dim_reduction__n_components': [10, 20, 30],
                'classifier__n_estimators': [50, 100, 150]
            }

        # Evaluate the combination
        accuracy, best_params = evaluate_model(pipeline, param_grid)
        results.append({
            'Dimensionality Reduction': dr_name,
            'Classifier': clf_name,
            'Test Accuracy': accuracy,
            'Best Params': best_params
        })

# Print results for dimensionality reduction + classification combinations
for result in results:
    print(f"Dimensionality Reduction: {result['Dimensionality Reduction']}, "
          f"Classifier: {result['Classifier']}, "
          f"Test Accuracy: {result['Test Accuracy']:.4f}, "
          f"Best Params: {result['Best Params']}")

# Now evaluate the classifiers on the original dataset (without dimensionality reduction)
print("\nRunning classifiers on original data (without dimensionality reduction)...")

# Load and preprocess the dataset
data = pd.read_csv('fashion-mnist_train-1.csv')
data.drop_duplicates(inplace=True)

X = data.drop('label', axis=1)
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=862)

for clf_name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classifier: {clf_name}, Test Accuracy (Original Data): {accuracy:.4f}")
