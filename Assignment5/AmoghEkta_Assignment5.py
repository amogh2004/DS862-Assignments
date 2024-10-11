#!/usr/bin/env python
# coding: utf-8

# # DS 862 - ASSIGNMENT 5
# ## AMOGH RANGANATHAIAH (aranganathaiah@sfsu.edu)
# ## EKTA SINGH (esingh@sfsu.edu)
# 
# For this assignment, you will use the German Credit Card data set from UCI data
# repository [(here)](https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29)

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv('gcredit.csv')
pd.set_option('display.max_columns', None)
data.head()


# In[3]:


pd.reset_option('display.max_columns')


# In[4]:


# Categorical Value
# Represents the status of the existing checking account.
data.V1.unique()


# In[5]:


# Numerical Value
# Represents the duration in months the account has been open.
data.V2


# In[6]:


# Categorical Value
# Represents the credit history, categorized into 5 different values (maybe poor, fair, good, very good, excellent)
data.V3.unique()


# In[7]:


# Categorical Value
# Represents the purpose of account
data.V4.unique()


# In[8]:


# Numerical Value
# Represents the credit amount associated with the account
data.V5


# In[9]:


# Categorical Value
# Represents the savings account or bonds status, categorized into different levels
data.V6.unique()


# In[10]:


# Categorical Value
# Represents the duration of present employment, categorized into different ranges
data.V7.unique()


# In[11]:


# Numerical Value
# Represents the installment rate as a percentage of disposable income
data.V8


# In[12]:


# Categorical Value
# Represents the marital status and sex of the individual, categorized into different groups
data.V9.unique()


# In[13]:


# Categorical Value
# Represents the presence of other debtors or guarantors, categorized into different types
data.V10.unique()


# In[14]:


# Numerical Value
# Represents the number of years the individual has lived at their present residence
data.V11


# In[15]:


# Categorical Value
# Represents the type of property owned, categorized into different types
data.V12.unique()


# In[16]:


# Numerical Value
# Represents the age of the individual in years
data.V13


# In[17]:


# Categorical Value
# Represents the presence of other installment plans, categorized into different types
data.V14.unique()


# In[18]:


# Categorical Value
# Represents the type of housing situation, categorized into different types
data.V15.unique()


# In[19]:


# Numerical Value
# Represents the number of existing credits the individual has at this bank
data.V16


# In[20]:


# Categorical Value
# Represents the individual's job or occupation, categorized into different types
data.V17.unique()


# In[21]:


# Numerical Value
# Represents the number of people the individual is financially liable to support
data.V18


# In[22]:


# Categorical Value(Binary)
# Indicates whether the individual has a registered telephone
data.V19.unique()


# In[23]:


# Categorical Value(Binary)
# Indicates whether the individual is a foreign worker
data.V20.unique()


# In[24]:


# Categorical Target Value(Binary)
# 1 = Good Customer, 2 = Bad Customer
data.V21.unique()


# **Numerical Features: V2, V5, V8, V11, V13, V16, V18** </br>
# **Categorical Features: V1, V3, V4, V6, V7, V9, V10, V12, V14, V15, V17, V19, V20**

# In[25]:


# Separate features (V1 to V20) and target (V21)
features = data.iloc[:, :-1]  # Select columns V1 to V20
target = data.iloc[:, -1]     # Select the target column V21

# Define numerical and categorical features
numerical_features = ['V2', 'V5', 'V8', 'V11', 'V13', 'V16', 'V18']
categorical_features = ['V1', 'V3', 'V4', 'V6', 'V7', 'V9', 'V10', 'V12', 'V14', 'V15', 'V17', 'V19', 'V20']

# Extract numerical and categorical feature subsets
numerical_data = features[numerical_features]
categorical_data = features[categorical_features]


# In[26]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Normalize numerical features using StandardScaler
scaler = StandardScaler()
scaled_numerical_data = scaler.fit_transform(numerical_data)

# # Apply MinMaxScaler
# scaler = MinMaxScaler()
# scaled_numerical_data = scaler.fit_transform(numerical_data)

# Convert back to DataFrame for consistency in concatenation if needed later
numerical_data = pd.DataFrame(scaled_numerical_data, columns=numerical_data.columns)


# In[27]:


from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode categorical features
encoded_categorical_data = categorical_data.apply(label_encoder.fit_transform)

# Encode the target variable (V21)
encoded_target = label_encoder.fit_transform(target)

# Combine numerical and encoded categorical features
encoded_features = pd.concat([numerical_data, encoded_categorical_data], axis=1)


# In[28]:


from sklearn.naive_bayes import GaussianNB

# Split the data into training and testing sets (numerical features only)
X_train, X_test, y_train, y_test = train_test_split(numerical_data, encoded_target, test_size=0.3, random_state=42)

# Initialize Gaussian Naive Bayes classifier
gnb = GaussianNB()

# Fit the model on the training set
gnb.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gnb.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# **Our Observation:** </br>
# GaussianNB is not performing well in this case. Numerical features alone are not capturing enough information to differentiate between good and bad customers, especially with the poor recall score. The model is weak at detecting "Bad" customers, which could be problematic for real-world applications.

# In[29]:


from sklearn.naive_bayes import CategoricalNB

# Split the categorical data into training and testing sets
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(encoded_categorical_data, encoded_target, test_size=0.3, random_state=42)

# Initialize Categorical Naive Bayes classifier
cnb = CategoricalNB()

# Fit the model on the training set
cnb.fit(X_train_cat, y_train_cat)

# Make predictions on the test set
y_pred_cat = cnb.predict(X_test_cat)

# Evaluate the model
accuracy_cat = accuracy_score(y_test_cat, y_pred_cat)
precision_cat = precision_score(y_test_cat, y_pred_cat)
recall_cat = recall_score(y_test_cat, y_pred_cat)
f1_cat = f1_score(y_test_cat, y_pred_cat)

# Print evaluation metrics
print(f"Accuracy: {accuracy_cat:.4f}")
print(f"Precision: {precision_cat:.4f}")
print(f"Recall: {recall_cat:.4f}")
print(f"F1 Score: {f1_cat:.4f}")


# **Our Observation:** </br>
# CategoricalNB performs better than GaussianNB, indicating that categorical features are more relevant for this problem. The improvement in recall and F1 score means the model is better at identifying "Bad" customers without too many false positives. This model could be more reliable for practical use, although there is still room for improvement in recall.

# In[30]:


# !pip install mixed-naive-bayes


# In[31]:


from mixed_naive_bayes import MixedNB

# Combine numerical and categorical features into a single dataset
combined_features = pd.concat([numerical_data, encoded_categorical_data], axis=1)

# Split the data into training and testing sets
X_train_mixed, X_test_mixed, y_train_mixed, y_test_mixed = train_test_split(combined_features, encoded_target, test_size=0.3, random_state=42)

# Initialize Mixed Naive Bayes classifier
mixed_nb = MixedNB()  # Replace with the actual constructor

# Fit the model on the training set
mixed_nb.fit(X_train_mixed, y_train_mixed)

# Make predictions on the test set
y_pred_mixed = mixed_nb.predict(X_test_mixed)

# Evaluate the model (using accuracy, precision, recall, F1-score)
accuracy_mixed = accuracy_score(y_test_mixed, y_pred_mixed)
precision_mixed = precision_score(y_test_mixed, y_pred_mixed)
recall_mixed = recall_score(y_test_mixed, y_pred_mixed)
f1_mixed = f1_score(y_test_mixed, y_pred_mixed)

# Print evaluation metrics
print(f"Accuracy: {accuracy_mixed:.4f}")
print(f"Precision: {precision_mixed:.4f}")
print(f"Recall: {recall_mixed:.4f}")
print(f"F1 Score: {f1_mixed:.4f}")


# **Our Observation:** </br>
# MixedNB shows a strong balance between precision and recall, with the highest recall and F1 score. This suggests that combining both numerical and categorical features provides more comprehensive information, improving the model's ability to identify "Bad" customers. While accuracy is slightly lower than CategoricalNB, the boost in recall makes this model more reliable for detecting risk, making it a better choice overall compared to the other two.

# **Overall Conclusion for ML Models:** </br>
# 1. GaussianNB struggles with detecting "Bad" customers, as shown by its very low recall and F1 score, making it less suitable for this problem. 
# 2. CategoricalNB performs better, especially in terms of recall and F1 score, suggesting categorical features play a key role in predicting customer classification.
# 3. MixedNB is the best performer overall, with the highest recall and F1 score. The combination of numerical and categorical features provides a better balance and improves the model's ability to correctly classify "Bad" customers.

# In[32]:


import pandas as pd

# Combine the training data (numerical and categorical) with the target
train_data_mixed = pd.concat([X_train_mixed, pd.Series(y_train_mixed, name='Target')], axis=1)

# Calculate the correlation matrix
correlation_matrix = train_data_mixed.corr(method='pearson')

# Display the correlation of each feature with the target
print(correlation_matrix['Target'])


# **General Observations:**
# 
# 1. Overall, none of the features exhibit strong correlations with the target variable. The highest absolute correlation is only 0.123 (for V6), indicating that no individual feature strongly predicts whether a customer will be classified as "Good" or "Bad."
# 
# 2. Given the weak correlations, feature engineering or transformation (such as interaction terms or non-linear relationships) might be required to better capture relationships in the data.
# 
# 3. We can consider applying advanced techniques like decision trees, random forests, or other non-linear models, as linear relationships (such as those reflected by Pearson correlation) are not well-captured in this dataset.

# In[33]:


import numpy as np
from scipy.stats import chi2_contingency

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    return np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))

# Apply Cramér’s V to each categorical feature in relation to the target
for col in categorical_data.columns:
    print(f"Cramér's V for {col} and Target: {cramers_v(X_train_mixed[col], y_train_mixed):.4f}")


# **General Observations:**
# 
# 1. V1 (Checking account status), V3 (Credit history), and V4 (Purpose of account) are the most relevant categorical features, as they show the strongest associations with the target variable. These factors likely contain useful information about the risk profile of the customers.
# 
# 2. Many other variables, especially those with Cramér’s V values below 0.1, have very weak associations with the target, suggesting they may not be very useful for the classification task on their own.
# 
# 3. Based on these values, you may want to focus on features like V1, V3, and V4 when building your model, or consider interactions with other features to strengthen predictive power.

# **Our Observations:**
# 
# 1. Checking account status represents the status of the individual’s existing checking account, which can reflect the financial health of the person. Checking account statuses often indicate whether there are positive balances, overdrafts, or negative balances.
# 
# 2. Credit history provides a summary of the individual's past credit behavior, including whether they have repaid previous debts on time or have a history of defaults, late payments, or delinquencies.
# 
# 3. The purpose of the account represents the reason or purpose for which the customer is using their account, such as taking out a loan for personal use, business, education, or purchasing a car or home.
# 
# **Why These Features Are Important in General:**
# 1. Risk Prediction: These features give direct insight into a customer's financial behavior and stability. The checking account status reflects liquidity, credit history reflects past behavior with credit, and the purpose of the account can reflect the riskiness of the customer’s spending or borrowing decisions.
# 
# 2. Behavioral Patterns: Understanding how a person has handled their finances in the past (credit history and checking account) and why they are requesting new financial products (account purpose) are key factors in predicting future behavior.
# 
# 3. Lender Decision-Making: In the banking and finance industry, lenders use these types of features to make informed decisions about offering credit, determining interest rates, and setting credit limits. Customers with a strong credit history and responsible account usage are more likely to be offered better terms, while those with poor histories may be offered stricter terms or rejected.
