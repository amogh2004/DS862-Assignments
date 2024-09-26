#!/usr/bin/env python
# coding: utf-8

# # DS 862 - ASSIGNMENT 3
# # AMOGH RANGANATHAIAH (aranganathaiah@sfsu.edu)
# # EKTA SINGH (esingh@sfsu.edu)

# For this assignment we will use the customer churning dataset and build a classifier for that. The data set is obtained from Kaggle [here](https://www.kaggle.com/blastchar/telco-customer-churn). The goal is to predict whether a customer will churn (i.e. leave) given a set of predictors.

# In[1]:


import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('churn.csv')
data.head()


# In[2]:


# Remove columns
data.drop(['customerID', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
'StreamingMovies', 'PaperlessBilling', 'PaymentMethod'], axis = 1, inplace = True)

# Create dummy variables
data = pd.get_dummies(data = data, columns = ['gender', 'Partner', 'Dependents',
'PhoneService', 'MultipleLines', 'InternetService', 'TechSupport', 'StreamingTV',
'Contract'], drop_first = True)
data.head()


# In[3]:


X = data.drop('Churn', axis = 1)
y = data.Churn


# Your task for this assignment is simple. First split your data into 80%-20%. Train a SVM with linear Kernel and investigate your classification error on the test set. Be sure to apply any appropriate preprocessing steps, and tune your model.

# In[4]:


# Convert 'Churn' column to numerical values
y = y.map({'No': 0, 'Yes': 1})


# In[5]:


import warnings
warnings.filterwarnings('ignore')


# In[6]:


# First split your data into 80%-20%. Split the data into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=862)


# In[7]:


from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Build a pipeline for scaling and training the SVM with linear kernel
svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_svc', LinearSVC(C=10, loss="squared_hinge", max_iter=1000, class_weight='balanced', random_state=862))
])

# Train the model
svm_clf.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_clf.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))


# Upon reaching this model, we also focused on tuning the hyperparameters. We employed techniques like GridSearchCV and RandomizedSearchCV to automatically identify the optimal combination of hyperparameters based on cross-validation. Additionally, we removed highly correlated features to enhance model performance. Despite these efforts, this model yielded the highest accuracy. We also attempted GridSearch using the F1-score as the evaluation metric, but the accuracy remained superior. Therefore, we will continue with this model before exploring polynomial and RBF kernels.

# In[8]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Create a pipeline for scaling and SVM with a polynomial kernel
svm_clf_poly = Pipeline([
    ('scaler', StandardScaler()),  # Feature scaling
    ('svm_poly', SVC(kernel='poly', random_state=862, class_weight='balanced'))  # Polynomial kernel SVM
])

# Define the parameter grid for GridSearchCV
param_grid_poly = {
    'svm_poly__C': [0.01, 0.1, 1, 10],  # Regularization parameter
    'svm_poly__degree': [2, 3, 4],  # Degrees of the polynomial kernel
    'svm_poly__coef0': [0, 1],  # Independent term in polynomial kernel
#     'svm_poly__gamma': ['scale', 'auto']  # Kernel coefficient
}

# Set up GridSearchCV to optimize hyperparameters with F1-score
grid_search_poly = GridSearchCV(svm_clf_poly, param_grid_poly, cv=5, scoring='f1')

# Fit the grid search to the training data
grid_search_poly.fit(X_train, y_train)

# Best parameters found by the grid search
print("Best parameters for polynomial kernel: ", grid_search_poly.best_params_)

# Best estimator
best_svm_clf_poly = grid_search_poly.best_estimator_

# Predict on the test set using the best model
y_pred_poly = best_svm_clf_poly.predict(X_test)

# Evaluate the model using accuracy and classification report
print(f"Accuracy: {accuracy_score(y_test, y_pred_poly)}")
print(classification_report(y_test, y_pred_poly))


# In[9]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Create a pipeline for scaling and SVM with an RBF kernel
svm_clf_rbf = Pipeline([
    ('scaler', StandardScaler()),  # Feature scaling
    ('svm_rbf', SVC(kernel='rbf', random_state=862, class_weight='balanced'))  # RBF kernel SVM
])

# Define the parameter grid for GridSearchCV
param_grid_rbf = {
    'svm_rbf__C': [0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'svm_rbf__gamma': ['scale', 'auto', 0.01, 0.001]  # Kernel coefficient (gamma)
}

# Set up GridSearchCV to optimize hyperparameters with F1-score
grid_search_rbf = GridSearchCV(svm_clf_rbf, param_grid_rbf, cv=5, scoring='f1')

# Fit the grid search to the training data
grid_search_rbf.fit(X_train, y_train)

# Best parameters found by the grid search
print("Best parameters for RBF kernel: ", grid_search_rbf.best_params_)

# Best estimator
best_svm_clf_rbf = grid_search_rbf.best_estimator_

# Predict on the test set using the best model
y_pred_rbf = best_svm_clf_rbf.predict(X_test)

# Evaluate the model using accuracy and classification report
print(f"Accuracy: {accuracy_score(y_test, y_pred_rbf)}")
print(classification_report(y_test, y_pred_rbf))


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Function to plot the confusion matrix
def plot_confusion_matrix(y_test, y_pred, title):
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churn', 'Churn'], yticklabels=['Not Churn', 'Churn'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

# Confusion matrix for Linear SVM
plot_confusion_matrix(y_test, y_pred, "Confusion Matrix - Linear Kernel SVM")

# Confusion matrix for Polynomial SVM
plot_confusion_matrix(y_test, y_pred_poly, "Confusion Matrix - Polynomial Kernel SVM")

# Confusion matrix for RBF SVM
plot_confusion_matrix(y_test, y_pred_rbf, "Confusion Matrix - RBF Kernel SVM")


# **General Observations:**
# 
# 1. True Negatives (TN):
# All three models perform similarly in predicting customers who did not churn (761 in Linear and Polynomial, and 754 in RBF), with a small drop in TN for the RBF model.
# 
# 2. False Positives (FP):
# The number of false positives is lowest for the Linear and Polynomial models (265) and slightly higher for the RBF model (272). This suggests that the RBF model tends to over-predict churn slightly more compared to the other models.
# 
# 3. False Negatives (FN):
# The Linear Kernel SVM misses the fewest churners (77), indicating it is better at identifying churn compared to the Polynomial (81) and RBF (90) models. The RBF model performs the worst in this regard, missing the most churners.
# 
# 4. True Positives (TP):
# The Linear Kernel SVM correctly identifies the most churners (304), followed by Polynomial (300), and RBF (291). This indicates that the Linear Kernel model has the highest sensitivity in identifying churn.
# 
# **Overall Conclusion:**
# 1. Linear Kernel SVM performs the best at identifying customers who churn, with the highest True Positives and the fewest False Negatives.
# 2. Polynomial Kernel SVM performs similarly to the Linear Kernel but with slightly worse recall (lower TP, higher FN).
# 3. RBF Kernel SVM has the worst performance in identifying churners, with both the highest False Negatives and slightly more False Positives, which indicates that it might not be the best model for this specific dataset.
# 
# <br> Thus, based on the confusion matrices, the Linear Kernel SVM seems to provide the best balance between precision and recall in this scenario.

# In[13]:


from sklearn.metrics import roc_curve, auc

# Function to plot the ROC curve
def plot_roc_curve(y_test, y_score, title):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


# In[14]:


# ROC Curve for Linear SVM
y_score_linear = svm_clf.named_steps['linear_svc'].decision_function(X_test)
plot_roc_curve(y_test, y_score_linear, "ROC Curve - Linear Kernel SVM")


# 1. ROC Curve Observation: The Linear Kernel SVM performs the worst among the three models, with an AUC score of 0.69. This indicates that the model's ability to distinguish between churn and non-churn customers is fairly limited.
# 2. True Positive Rate (Sensitivity): The True Positive Rate is relatively low across a large range of False Positive Rates, meaning the model struggles to accurately predict churn while keeping false positives low.
# 3. Overall: With an AUC of 0.69, the model is not highly effective at distinguishing between the two classes, suggesting that the linear decision boundary may not be sufficient for this problem.

# In[15]:


# ROC Curve for Polynomial SVM
y_score_poly = best_svm_clf_poly.decision_function(X_test)
plot_roc_curve(y_test, y_score_poly, "ROC Curve - Polynomial Kernel SVM")


# 1. ROC Curve Observation: The Polynomial Kernel SVM performs the best, with an AUC score of 0.83, indicating good discrimination between churn and non-churn customers.
# 2. True Positive Rate (Sensitivity): The curve rises steeply, indicating that the model has a much better True Positive Rate at relatively low False Positive Rates compared to the Linear Kernel. This suggests that the Polynomial Kernel is capturing more complex relationships between the features.
# 3. Overall: With an AUC of 0.83, the Polynomial Kernel SVM is the most effective model at distinguishing between churn and non-churn customers.

# In[16]:


# ROC Curve for RBF SVM
y_score_rbf = best_svm_clf_rbf.decision_function(X_test)
plot_roc_curve(y_test, y_score_rbf, "ROC Curve - RBF Kernel SVM")


# 1. ROC Curve Observation: The RBF Kernel SVM has an AUC of 0.82, which is quite close to the Polynomial Kernel SVM's performance.
# 2. True Positive Rate (Sensitivity): Similar to the Polynomial Kernel, the RBF Kernel demonstrates a good ability to capture the relationships in the data, with a strong True Positive Rate at low False Positive Rates. However, its curve is slightly less steep than the Polynomial Kernel.
# 3. Overall: The RBF Kernel SVM is also quite effective at distinguishing between the classes, though it performs slightly worse than the Polynomial Kernel.

# ## Further Observations
# 
# 1. The performance of non-linear kernels (Polynomial and RBF) indicates that there may be complex interactions between features that the Linear Kernel is unable to capture.
# 2. Non-linear models like Polynomial and RBF kernels perform better, indicating that the relationships between variables are complex. Trying using ensemble methods (like stacking or boosting) could improve performance over any individual classifier.

# ## Conclusion
# 
# 1. Polynomial Kernel SVM performs the best, with an AUC of 0.83, followed closely by the RBF Kernel SVM (AUC = 0.82).
# 2. The Linear Kernel SVM lags behind with an AUC of 0.69, indicating that it struggles to capture the complexity of the data compared to the non-linear kernels (Polynomial and RBF).
# 3. Both the Polynomial and RBF kernels significantly outperform the Linear Kernel, highlighting that non-linear decision boundaries are more suited for this churn prediction problem.
# 4. The Polynomial and RBF kernel models are performing well, so focusing on improving the recall of churned customers, understanding feature importance, and considering business-specific metrics will help maximize the model’s usefulness in practical applications. We could tune this model, if we knew exactly what the business-specific metrics would be.
