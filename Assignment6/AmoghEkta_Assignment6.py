#!/usr/bin/env python
# coding: utf-8
# Group Name:Amogh Ranganathaih ; Ekta Singh
# In[1]:


# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.svm import SVC
get_ipython().system('brew install libomp')
get_ipython().system('pip install xgboost')
get_ipython().system('pip install lightgbm')
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


# In[2]:


# Load the dataset
data = pd.read_csv('/Users/vega/Desktop/bank_churn.csv')

data.head()


# In[3]:


# Preprocessing: Drop unnecessary columns
data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Convert categorical variables to dummy variables
data = pd.get_dummies(data, drop_first=True)

# Define X and y
X = data.drop('Exited', axis=1)  # Features
y = data['Exited']  # Target


# In[4]:


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=862)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[5]:


# Defining classifiers
knn_clf1 = KNeighborsClassifier()  # First KNN classifier
knn_clf2 = KNeighborsClassifier()  # Second KNN classifier
dt_clf = DecisionTreeClassifier(random_state=862)
rf_clf = RandomForestClassifier(random_state=862)
svc_clf1 = SVC(probability=True, random_state=862)
svc_clf2 = SVC(probability=True, random_state=862)

# Tuning individual models using GridSearchCV
param_grid_knn = {'n_neighbors': [5, 7, 10]}
grid_knn1 = GridSearchCV(knn_clf1, param_grid_knn, cv=5)
grid_knn2 = GridSearchCV(knn_clf2, param_grid_knn, cv=5)

param_grid_dt = {'max_depth': [None, 5, 10, 15]}
grid_dt = GridSearchCV(dt_clf, param_grid_dt, cv=5)

param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [None, 5, 10]}
grid_rf = GridSearchCV(rf_clf, param_grid_rf, cv=5)

param_grid_svc1 = {'C': [0.1, 1, 10], 'kernel': ['rbf']}
param_grid_svc2 = {'C': [1, 10, 100], 'kernel': ['rbf']}

grid_svc1 = GridSearchCV(svc_clf1, param_grid_svc1, cv=5)
grid_svc2 = GridSearchCV(svc_clf2, param_grid_svc2, cv=5)

# Fitting the models
grid_knn1.fit(X_train, y_train)
grid_knn2.fit(X_train, y_train)
grid_dt.fit(X_train, y_train)
grid_rf.fit(X_train, y_train)
grid_svc1.fit(X_train, y_train)
grid_svc2.fit(X_train, y_train)

# Best models
knn_best1 = grid_knn1.best_estimator_
knn_best2 = grid_knn2.best_estimator_
dt_best = grid_dt.best_estimator_
rf_best = grid_rf.best_estimator_
svc_best1 = grid_svc1.best_estimator_
svc_best2 = grid_svc2.best_estimator_


# In[6]:


# Soft Voting Classifier
voting_clf_soft = VotingClassifier(estimators=[
    ('knn1', knn_best1), ('knn2', knn_best2), 
    ('dt', dt_best), ('rf', rf_best), 
    ('svc1', svc_best1)], voting='soft')

voting_clf_soft.fit(X_train, y_train)
y_pred_soft = voting_clf_soft.predict(X_test)
print("Soft Voting Classifier Accuracy:", accuracy_score(y_test, y_pred_soft))


# In[7]:


# Hard Voting Classifier
voting_clf_hard = VotingClassifier(estimators=[
    ('knn1', knn_best1), ('knn2', knn_best2), 
    ('dt', dt_best), ('rf', rf_best), 
    ('svc1', svc_best1)], voting='hard')

voting_clf_hard.fit(X_train, y_train)
y_pred_hard = voting_clf_hard.predict(X_test)
print("Hard Voting Classifier Accuracy:", accuracy_score(y_test, y_pred_hard))


# In[8]:


#The soft voting classifier, which uses the predicted probabilities from each model, performed quite well, achieving an accuracy of 80.6%. This suggests a good level of confidence in model predictions through weighted voting compared to the non-weighted method of using hot voting classifier which is giving an accuracy of 79%.It shows the effectiveness of combined weight possibilities.


# In[9]:


# Compare individual models
print("Best KNN (5 neighbors) Accuracy:", accuracy_score(y_test, knn_best1.predict(X_test)))
print("Best KNN (7 neighbors) Accuracy:", accuracy_score(y_test, knn_best2.predict(X_test)))
print("Best Decision Tree Accuracy:", accuracy_score(y_test, dt_best.predict(X_test)))
print("Best Random Forest Accuracy:", accuracy_score(y_test, rf_best.predict(X_test)))
print("Best SVC1 Accuracy:", accuracy_score(y_test, svc_best1.predict(X_test)))
print("Best SVC2 Accuracy:", accuracy_score(y_test, svc_best2.predict(X_test)))


# In[10]:


#I have tuned and fit the models and hence now Random forest gives the highest accuracy of 85.45% which is higher than soft voting method, followed by decision tree accuracy of 84.45%.It would be lesser had the tuning not been done.


# In[11]:


# Create BaggingClassifier with SVM as the base estimator
bagged_svm = BaggingClassifier(base_estimator=SVC(probability=True, random_state=862), 
                               n_estimators=100, oob_score=True)

# Fit the Bagged SVM model
bagged_svm.fit(X_train, y_train)

# Make predictions on the test set
y_pred_bagged = bagged_svm.predict(X_test)

# Calculate accuracy
bagged_accuracy = accuracy_score(y_test, y_pred_bagged)
print("Bagged SVM Accuracy:", bagged_accuracy)

# Evaluate out-of-bag score
oob_score = bagged_svm.oob_score_
print("Out-of-Bag Score for Bagged SVM:", oob_score)



# In[12]:


#The Bagged SVM model achieved an accuracy of 78.75%, which is comparable to the individual SVC models. The out-of-bag score of 79.85% indicates that the model generalizes well, but it did not significantly outperform the standalone SVM models.It can be improved by tuning the parameters further to give a better accuracy.


# In[14]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Create an XGBoost classifier
xgboost_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', 
                            n_estimators=100, max_depth=3, learning_rate=0.1, 
                            random_state=862, n_jobs=-1)

# Fit the model on training data
xgboost_clf.fit(X_train_scaled, y_train)

# Evaluate the model
accuracy_xgboost = xgboost_clf.score(X_test_scaled, y_test)
print(f'XGBoost Accuracy: {accuracy_xgboost:.4f}')


# In[15]:


#The XGBoost classifier achieved a higher accuracy of 0.8495, outperforming the Bagged SVM, which had an accuracy of 0.7875 and an Out-of-Bag score of 0.7985. This suggests that XGBoost is more effective than Bagging with SVM for this dataset, demonstrating superior predictive performance.


# In[16]:


import lightgbm as lgb
from lightgbm import LGBMClassifier

# Create a LightGBM classifier
lgbm_clf = LGBMClassifier(num_leaves=31, learning_rate=0.01, 
                          n_estimators=100, random_state=862)

# Fit the model on training data
lgbm_clf.fit(X_train_scaled, y_train, 
             eval_set=[(X_test_scaled, y_test)], 
             eval_metric='logloss', 
             callbacks=[lgb.early_stopping(stopping_rounds=5)])

# Evaluate the model
accuracy_lgbm = lgbm_clf.score(X_test_scaled, y_test)
print(f"LightGBM Classifier Accuracy: {accuracy_lgbm:.4f}")


# In[17]:


#Both LightGBM and XGBoost classifiers achieved an identical accuracy of 0.8495 on the test set. This indicates that both gradient-boosting methods are performing equally well for the given dataset, suggesting minimal differences in their effectiveness for this classification task under the current model configurations.


# In[18]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
# Define base learners
models = {
    'rf': RandomForestClassifier(n_estimators=100, random_state=862),
    'ada': AdaBoostClassifier(n_estimators=100, random_state=862),
    'gb': GradientBoostingClassifier(n_estimators=100, random_state=862),
    'svc': SVC(probability=True, random_state=862)
}

# Define the blender
blender = LogisticRegression()

# Split the training data into two parts
X_train1, X_train2, y_train1, y_train2 = train_test_split(X_train_scaled, y_train, test_size=0.5, random_state=862)

# Train the weak learners
for name, model in models.items():
    model.fit(X_train1, y_train1)

# Train the blender
# Get predictions from the weak learners
predictions = pd.DataFrame()
for name, model in models.items():
    predictions[name] = model.predict(X_train2)

# Scale the predictions for the blender
scaler_blend = StandardScaler()
predictions_scaled = scaler_blend.fit_transform(predictions)

# Fit the blender model
blender.fit(predictions_scaled, y_train2)

# Evaluate the stacking model

# Get predictions from the weak learners on the test set
predictions_test = pd.DataFrame()
for name, model in models.items():
    predictions_test[name] = model.predict(X_test_scaled)

# Scale the predictions for the test set
predictions_test_scaled = scaler_blend.transform(predictions_test)

# Get final predictions from the blender
final_predictions = blender.predict(predictions_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, final_predictions)
print("Stacking Classifier Accuracy:", accuracy)


# In[19]:


#The stacking classifier performed as well as the best individual model (Random Forest) with an accuracy of 85.45%. By combining predictions from diverse base learners (RandomForest, AdaBoost, GradientBoosting, SVC), the stacking approach leveraged their strengths but did not outperform the Random Forest. It shows that stacking effectively generalizes but adds complexity without significant accuracy gains in this case.


# In[ ]:





# In[ ]:





# In[ ]:




