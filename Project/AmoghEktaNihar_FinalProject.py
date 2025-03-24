#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv('credit_score_train.csv')
pd.set_option('display.max_columns', None)


# In[3]:


data


# ### Data Preprocessing

# In[4]:


data.drop(['id', 'customer_id', 'month', 'name', 'ssn', 'occupation', 'type_of_loan', 'payment_behaviour'], axis=1, inplace=True)


# **Columns Dropped**
# 
# 1. id: This column contains a unique identifier for each record, which has no predictive value for credit_score. Retaining it would add noise and potentially confuse models.
# 2. customer_id: Similar to id, it is a unique identifier for customers. It does not contribute to understanding or predicting credit_score. Including it would lead to overfitting and no generalizable insights.
# 3. month: The correlation between month and credit_score is extremely low (0.01), and the dataset spans only one year. Hence, it lacks significance. Including it would add unnecessary complexity without improving model performance.
# 4. name: Names are purely identifiers and contain no information relevant to credit_score. Keeping it might lead to unintended bias in the model, and it has no analytical value.
# 5. ssn: Social Security Numbers (SSNs) are unique identifiers that are irrelevant for predicting credit scores. Retaining sensitive personally identifiable information (PII) like SSNs is a data privacy risk.
# 6. occupation: While occupation may have some correlation with income or credit behavior, it often leads to overgeneralization or stereotyping. Furthermore, it may already be indirectly captured through other features like annual_income or monthly_inhand_salary. Including it can complicate the model without significantly improving predictive power.
# 7. type_of_loan: This column contains complex, multi-value data (e.g., a list of loan types). It would require significant preprocessing, and its value might already be represented by features like num_of_loan or credit_mix. It adds redundancy and complexity.
# 8. payment_behaviour: This column seems to categorize spending and payment patterns (e.g., High_spent_Small_value_payments). While potentially insightful, it may overlap with features like credit_utilization_ratio or payment_of_min_amount. Including it without detailed preprocessing could lead to redundancy or noise.

# In[5]:


data.info()


# In[6]:


# 0 => Poor
# 1 => Standard
# 2 => Good

data['credit_score'].value_counts()


# In[7]:


data['credit_mix'].value_counts()


# In[8]:


credit_mix_map = {'Bad': 0, 'Standard': 1, 'Good': 2}
data['credit_mix'] = data['credit_mix'].map(credit_mix_map)


# In[9]:


data['payment_of_min_amount'].value_counts()


# In[10]:


min_amount_map = {'No': 0, 'Yes': 1}
data['payment_of_min_amount'] = data['payment_of_min_amount'].map(min_amount_map)


# In[11]:


# Check the number of unique values in each column
for col in data.columns:
    print(f"{col}: {data[col].nunique()} unique values")


# In[12]:


data['num_bank_accounts'].value_counts()


# In[13]:


data['num_credit_card'].value_counts()


# In[14]:


data['num_of_loan'].value_counts()


# To optimize the number of categories for num_bank_accounts, num_of_loan and num_credit_card while implementing a classifier for credit_score, we are grouping the values into bins that reduce sparsity and improve model generalization. 
# 
# 1. **num_bank_accounts**
#     - Most values are concentrated between 3 and 10.
#     - Sparse categories are 0, 1, 2, and 11.
#     - Binning Strategy: </br>
#         Low (0–2): 0, 1, 2 (Rare, can be grouped together). </br>
#         Moderate (3–5): 3, 4, 5 (Most common range). </br>
#         High (6–8): 6, 7, 8 (Next most common range). </br>
#         Very High (9+): 9, 10, 11 (Sparse high values grouped together). </br> </br>
#         
# 2. **num_of_loan**
#     - Most values are concentrated between 0 and 4.
#     - Sparse categories are 8 and 9.
#     - Binning Strategy: </br>
#         None (0): 0 (Distinct case with no loans). </br>
#         Very Low (1–2): 1, 2 (Common low values). </br>
#         Low (3–4): 3, 4 (Next most common range). </br>
#         Moderate (5–6): 5, 6 (Moderate range). </br>
#         High (7+): 7, 8, 9 (Sparse high values grouped together). </br> </br>
#         
# 3. **num_credit_card**
#     - Most values are between 3 and 8.
#     - Sparse categories are 0, 1, 2, 10, and 11.
#     - Binning Strategy: </br>
#         None (0): 0 (Distinct case). </br>
#         Very Low (1–2): 1, 2 (Rare values grouped). </br>
#         Low (3–4): 3, 4 (Common range). </br>
#         Moderate (5–7): 5, 6, 7 (Most common range). </br>
#         High (8+): 8, 9, 10, 11 (Sparse high values grouped). </br>

# In[15]:


# Low => 0
# Moderate => 1
# High => 2
# Very High => 3

# Define bins and labels for num_bank_accounts
bins_bank_accounts = [0, 2, 5, 8, float('inf')]
labels_bank_accounts = [0, 1, 2, 3]
data['num_bank_accounts'] = pd.cut(data['num_bank_accounts'], bins=bins_bank_accounts, labels=labels_bank_accounts, right=True).astype(float)


# In[16]:


# None => 0
# Very Low => 1
# Low => 2
# Moderate => 3
# High => 4

# Define bins and labels for num_of_loan
bins_num_of_loan = [-1, 0, 2, 4, 6, float('inf')]
labels_num_of_loan = [0, 1, 2, 3, 4]

# Create a new binned column for num_of_loan
data['num_of_loan'] = pd.cut(data['num_of_loan'], bins=bins_num_of_loan, labels=labels_num_of_loan, right=True).astype(float)


# In[17]:


# None => 0
# Very Low => 1
# Low => 2
# Moderate => 3
# High => 4

# Define bins and labels for num_credit_card
bins_num_credit_card = [-1, 0, 2, 4, 7, float('inf')]
labels_num_credit_card = [0, 1, 2, 3, 4]

# Create a new binned column for num_credit_card
data['num_credit_card'] = pd.cut(data['num_credit_card'], bins=bins_num_credit_card, labels=labels_num_credit_card, right=True).astype(float)


# In[18]:


# month_mapping = {
#     'January': 1, 'February': 2, 'March': 3, 'April': 4,
#     'May': 5, 'June': 6, 'July': 7, 'August': 8,
#     'September': 9, 'October': 10, 'November': 11, 'December': 12
# }

# # Map the month names to numerical values
# data['month'] = data['month'].map(month_mapping)

# print(data.corr()['credit_score'])


# In[19]:


data.dropna(inplace=True)


# In[20]:


# Convert the specified columns to integer type
columns_to_convert = ['num_bank_accounts', 'num_credit_card', 'num_of_loan']

for col in columns_to_convert:
    data[col] = data[col].astype(int)


# In[21]:


data.info()


# In[22]:


data


# In[23]:


# Define the target variable
target_variable = 'credit_score'

# Separate categorical columns
categorical_columns = [
    'num_bank_accounts', 'num_credit_card', 'num_of_loan', 'credit_mix', 'payment_of_min_amount'
]

# Separate numerical columns
numerical_columns = [
    'age', 'annual_income', 'monthly_inhand_salary', 'credit_history_age', 'total_emi_per_month', 'interest_rate',
    'delay_from_due_date', 'num_of_delayed_payment', 'changed_credit_limit', 'num_credit_inquiries', 
    'outstanding_debt', 'credit_utilization_ratio', 'amount_invested_monthly', 'monthly_balance'
]


# In[24]:


from scipy.stats import chi2_contingency

# Function to perform chi-square test
def chi_square_test(data, categorical_columns, target_variable):
    results = {}
    for col in categorical_columns:
        # Create a contingency table
        contingency_table = pd.crosstab(data[col], data[target_variable])
        
        # Perform the chi-square test
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        
        # Store the results
        results[col] = {'chi2': chi2, 'p-value': p, 'dof': dof}
    
    # Convert results to a DataFrame for easy interpretation
    results_df = pd.DataFrame(results).T
    return results_df

# Perform the chi-square test
chi_square_test(data, categorical_columns, target_variable)


# In[25]:


from scipy.stats import f_oneway

# Function to perform ANOVA test
def anova_test(data, numerical_columns, target_variable):
    results = {}
    for col in numerical_columns:
        # Group data by the target variable
        groups = [data[col][data[target_variable] == category] for category in data[target_variable].unique()]
        
        # Perform the ANOVA test
        f_stat, p_value = f_oneway(*groups)
        
        # Store the results
        results[col] = {'F-statistic': f_stat, 'p-value': p_value}
    
    # Convert results to a DataFrame for easy interpretation
    results_df = pd.DataFrame(results).T
    return results_df

anova_test(data, numerical_columns, target_variable)


# All categorical variables show a significant relationship with credit_score and should be considered for modeling.
# 1. num_bank_accounts
# 2. num_credit_card
# 3. num_of_loan
# 4. credit_mix
# 5. payment_of_min_amount
# 
# The numerical columns with high F-statistics and very small p-values which are likely to play a critical role in determining **credit_score** are:
# 1. interest_rate
# 2. credit_history_age
# 3. delay_from_due_date
# 4. num_credit_inquiries
# 5. num_of_delayed_payment
# 6. outstanding_debt
# 
# Variables like **total_emi_per_month** and **changed_credit_limit** have moderate F-statistics but are still statistically significant. </br>
# 
# **credit_utilization_ratio** and **amount_invested_monthly** have relatively low F-statistics, suggesting they have less impact on credit_score. </br>

# ### Hypothesis Testing Conclusion
# Based on the ANOVA test results, the null hypothesis (H0) that changes in credit card limits (changed_credit_limit) have no significant effect on credit scores across categories is rejected. The test showed a statistically significant difference in the mean changed_credit_limit across credit score categories (p-value = 0.000). However, despite this statistical significance, the relative importance of changed_credit_limit as a predictor for credit score is low compared to other variables. Therefore, changed_credit_limit will not be included in the data modeling process.

# ### Principal Component Analysis
# 
# Given the manageable feature size and the importance of interpretability in a financial context, PCA may not add significant value. Models like Decision Trees, Random Forests, and Gradient Boosting can naturally handle correlated and redundant features. Instead, we went ahead with Feature Importance Analysis.
# 
# ### Feature Importance Analysis
# Feature importance analysis is a critical step in the modeling process, particularly for models like **Decision Trees**, **Random Forests**, and **Gradient Boosting (CatBoost)**, as these models inherently calculate the contribution of each feature to the prediction. By analyzing feature importance:
# 1. Improved Interpretability: It provides valuable insights into which features have the greatest influence on the target variable, enabling better understanding and explanation of the model's behavior.
# 2. Enhanced Model Performance: Identifying and prioritizing important features allows for the removal of irrelevant or redundant ones, reducing overfitting and improving generalization.
# 3. Efficient Resource Utilization: By focusing on the most influential features, computational efficiency is improved, which is particularly beneficial when working with high-dimensional datasets.
# 
# For **Neural Networks**, feature importance analysis using methods like **SHAP** or **Integrated Gradients** helps address the black-box nature of these models, making their predictions more transparent. 
# 
# While **K-Nearest Neighbors (KNN) does not inherently compute feature importance**, alternative methods like **permutation importance** can provide insights if needed.
# 
# In this project, feature importance analysis is integral for building interpretable and robust models while ensuring the derived insights align with the domain-specific understanding of the data.

# ### Data Visualizations

# In[26]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a bar chart
plt.figure(figsize=(6, 4))
plt.bar(data['credit_score'].value_counts().index, data['credit_score'].value_counts().values)
plt.xlabel('Credit Score')
plt.ylabel('Frequency')
plt.title('Distribution of Credit Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[27]:


# Get the count of each credit mix category
credit_mix_counts = data['credit_mix'].value_counts()

# Create a bar chart
plt.figure(figsize=(6, 4))
plt.bar(credit_mix_counts.index, credit_mix_counts.values)
plt.xlabel('Credit Mix')
plt.ylabel('Count')
plt.title('Distribution of Credit Mix')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Ensure labels fit within the figure area
plt.show()


# In[28]:


# Group by credit score and calculate the average delay from due date
df_grouped = data.groupby('credit_score')['delay_from_due_date'].mean().reset_index()

# Create a bar chart
plt.figure(figsize=(6, 4))
plt.bar(df_grouped['credit_score'], df_grouped['delay_from_due_date'])
plt.xlabel('Credit Score')
plt.ylabel('Average Delay from Due Date')
plt.title('Delay from Due Date vs. Credit Score')
plt.show()


# In[29]:


# Create a bar chart
plt.figure(figsize=(6, 4))
data['payment_of_min_amount'].value_counts().plot(kind='bar')
plt.xlabel('Payment Status')
plt.ylabel('Frequency')
plt.title('Payment of Minimum Amount', fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[30]:


# Define the numerical features and the target variable
numerical_features = [
    'interest_rate', 'credit_history_age', 'delay_from_due_date', 
    'num_credit_inquiries', 'num_of_delayed_payment', 'outstanding_debt'
]
target = 'credit_score'

# Create a subset of the data with numerical features and the target
subset_data = data[numerical_features + [target]]

# Compute the correlation matrix
correlation_matrix = subset_data.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features and Credit Score")
plt.show()


# In[31]:


import scipy.stats as ss

# Function to compute Cramér's V
def cramers_v(confusion_matrix):
    """Calculate Cramér's V statistic for categorical-categorical association."""
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.values.sum()  # Use .values.sum() to ensure scalar
    r, k = confusion_matrix.shape

    # To avoid division by zero or ambiguous calculations
    if n == 0:
        return 0.0

    phi2 = chi2 / n
    phi2corr = max(0, phi2 - ((k-1)*(r-1)) / max((n-1), 1))  # Ensure denominator > 0
    rcorr = max(r - ((r-1)**2) / max((n-1), 1), 1)
    kcorr = max(k - ((k-1)**2) / max((n-1), 1), 1)
    
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# Function to compute Cramér's V matrix
def compute_cramers_v_matrix(df, columns):
    """Compute pairwise Cramér's V for a set of categorical columns."""
    n = len(columns)
    cramers_v_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                cramers_v_matrix[i, j] = 1.0  # Perfect correlation on diagonal
            else:
                confusion_matrix = pd.crosstab(df[columns[i]], df[columns[j]])
                cramers_v_matrix[i, j] = cramers_v(confusion_matrix)
    return cramers_v_matrix

columns = categorical_columns + [target]

# Replace 'data' with your DataFrame
cramers_v_matrix = compute_cramers_v_matrix(data, columns)

# Convert to DataFrame for visualization
cramers_v_df = pd.DataFrame(cramers_v_matrix, index=columns, columns=columns)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cramers_v_df, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Cramér's V Correlation Heatmap for Categorical Features")
plt.show()


# ### Splitting the data into Train and Validation

# While determining feature importance during data modeling, **payment_of_min_amount** was found to be **insignificant** for all models, and thus it was excluded from further consideration.

# In[32]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

# Define the features (independent variables) and target variable
features = [
    'num_bank_accounts', 'num_credit_card', 'num_of_loan', 'credit_mix', 
    'interest_rate', 'credit_history_age', 'delay_from_due_date', 
    'num_credit_inquiries', 'num_of_delayed_payment', 'outstanding_debt'
]
target = 'credit_score'

# Split the data into training and validation sets (80% train, 20% validation)
X = data[features]  # Feature matrix
y = data[target]    # Target variable

# stratify=y => Ensures 'credit_score' is proportionally represented in both training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[33]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# In machine learning, **non-binary classification models** refers to any model capable of predicting more than two categories, typically called **multi-class classification models**, which include algorithms like: 
# 1. Decision Trees
# 2. K-Nearest Neighbors (KNN)
# 3. Random Forests
# 4. Gradient Boosting(CatBoost)
# 5. Neural Networks
# 
# all of which can be adapted to classify data into multiple distinct classes beyond just "yes" or "no". We will be implementing all the above algorithms and comparing the results to identify the best performing model.

# ### Decision Trees

# In[34]:


# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import GridSearchCV

# # Define the model
# dt_model = DecisionTreeClassifier(random_state=42)

# # Define the hyperparameters to tune
# param_grid = {
#     'criterion': ['gini', 'entropy'],  # Splitting criteria
#     'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
#     'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
#     'min_samples_leaf': [1, 2, 5]     # Minimum samples required to be a leaf node
# }

# # Perform grid search with cross-validation
# grid_search = GridSearchCV(
#     estimator=dt_model,
#     param_grid=param_grid,
#     scoring='accuracy',
#     cv=5,
#     verbose=1,
#     n_jobs=-1
# )

# # Fit the model on the training data
# grid_search.fit(X_train, y_train)

# # Get the best model
# best_dt_model = grid_search.best_estimator_

# # Evaluate the model on the validation set
# y_pred = best_dt_model.predict(X_val)

# # Print results
# print("Best Parameters:", grid_search.best_params_)
# print("Training Accuracy:", grid_search.best_score_)
# print("Validation Accuracy:", accuracy_score(y_val, y_pred))
# print("\nClassification Report:\n", classification_report(y_val, y_pred))


# In[35]:


from sklearn.tree import DecisionTreeClassifier

# Define the model with the best parameters
best_dt_model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=30,
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=42
)

# Train the model on the training data
best_dt_model.fit(X_train, y_train)

# Predict on the validation set
y_pred = best_dt_model.predict(X_val)

# Calculate and print validation accuracy
val_accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy for Decision Tree Classifier: {val_accuracy:.2f}")

# Optionally, print the classification report
print("\nClassification Report:\n", classification_report(y_val, y_pred))


# In[36]:


# Feature Importance Analysis
feature_importances = best_dt_model.feature_importances_

# Create a DataFrame for visualization
importances_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plot the feature importances with values displayed on the bars
plt.figure(figsize=(10, 6))
bars = plt.bar(importances_df['Feature'], importances_df['Importance'], color='skyblue')

# Annotate each bar with the importance value
for bar, importance in zip(bars, importances_df['Importance']):
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # X position (center of the bar)
        bar.get_height(),                   # Y position (height of the bar)
        f'{importance:.2f}',                # Importance value formatted to 2 decimal points
        ha='center', va='bottom', fontsize=10  # Center alignment and font size
    )

plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances for Decision Tree Classifier')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# We attempted to remove *num_credit_card*, *num_of_loan*, *num_bank_accounts*, and *payment_of_min_amount* due to their minimal importance as features. However, this did not improve the model's fit.
# 
# **Classification Report**
# 1. Class 1 performs the best with an F1-score of 0.79, indicating balanced precision and recall.
# 2. Class 2 has the lowest performance with an F1-score of 0.70, likely due to fewer samples in the validation set (support=3151).
# 3. Class 0 performs moderately with an F1-score of 0.76.
# 4. The macro average F1-score (0.75) indicates balanced performance across classes.
# 5. The weighted average F1-score (0.77) suggests that the model performs better for more frequent classes (e.g., class 1).
# 
# **Model Suitability**
# 1. Decision Trees are interpretable, making them a good fit for understanding feature contributions in this financial dataset.
# 2. However, the model might overfit due to the tree's inherent nature of splitting data finely.
# 
# **Business Implications**
# 1. Insights for Credit Scoring:
#     - Outstanding Debt: This feature's high importance aligns with the business understanding that higher debt levels are crucial in determining credit risk.
#     - Credit History Age: The strong contribution of this feature emphasizes the importance of long-term credit usage in assessing creditworthiness.
#     - Credit Mix: Diversification of credit types appears to be a key determinant of credit scores, reflecting a consumer's ability to handle different financial products.
# 
# 2. Decision-Making Recommendations:
#     - Focus on Key Features: Prioritize outstanding_debt and credit_history_age in policy-making and customer risk assessment strategies.
# 
# 3. Actionable Insights:
#     - Tailor financial products based on outstanding debt levels and credit history.
#     - Educate consumers about the importance of maintaining a healthy credit mix and consistent payment behavior to improve their credit scores.

# ### K-Nearest Neighbors (KNN)

# In[37]:


# from sklearn.neighbors import KNeighborsClassifier

# # Define the model
# knn_model = KNeighborsClassifier()

# # Define the hyperparameters to tune
# param_grid = {
#     'n_neighbors': [3, 5, 7, 9, 11],  # Number of neighbors to consider
#     'weights': ['uniform', 'distance'],  # Weight function used in prediction
#     'metric': ['euclidean', 'manhattan', 'minkowski']  # Distance metric
# }

# # Perform grid search with cross-validation
# grid_search = GridSearchCV(
#     estimator=knn_model,
#     param_grid=param_grid,
#     scoring='accuracy',
#     cv=5,
#     verbose=1,
#     n_jobs=-1
# )

# # Fit the model on the training data
# grid_search.fit(X_train, y_train)

# # Get the best model
# best_knn_model = grid_search.best_estimator_

# # Evaluate the model on the validation set
# y_pred = best_knn_model.predict(X_val)

# # Print results
# print("Best Parameters:", grid_search.best_params_)
# print("Training Accuracy:", grid_search.best_score_)
# print("Validation Accuracy:", accuracy_score(y_val, y_pred))
# print("\nClassification Report:\n", classification_report(y_val, y_pred))


# In[38]:


from sklearn.neighbors import KNeighborsClassifier

# Define the model with the best parameters
best_knn_model = KNeighborsClassifier(
    metric='manhattan',
    n_neighbors=9,
    weights='distance'
)

# Train the model on the training data
best_knn_model.fit(X_train, y_train)

# Predict on the validation set
y_pred = best_knn_model.predict(X_val)

# Calculate and print validation accuracy
val_accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy for K-Nearest Neighbors Classifier: {val_accuracy:.2f}")

# Optionally, print the classification report
print("\nClassification Report:\n", classification_report(y_val, y_pred))


# In[39]:


from sklearn.inspection import permutation_importance

# Calculate permutation importance
result = permutation_importance(
    best_knn_model, X_val, y_val, scoring='accuracy', n_repeats=10, random_state=42
)

# Extract feature importances and corresponding feature names
feature_importances = result.importances_mean
feature_names = [
    'num_bank_accounts', 'num_credit_card', 'num_of_loan', 'credit_mix', 
    'interest_rate', 'credit_history_age', 'delay_from_due_date', 
    'num_credit_inquiries', 'num_of_delayed_payment', 'outstanding_debt'
]
# Create a DataFrame for visualization
importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display the feature importances
# print("\nPermutation Feature Importances:\n", importances_df)

# Plot the feature importances
plt.figure(figsize=(10, 6))
bars = plt.bar(importances_df['Feature'], importances_df['Importance'], color='skyblue')

# Annotate each bar with the importance value
for bar, importance in zip(bars, importances_df['Importance']):
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # X position (center of the bar)
        bar.get_height(),                   # Y position (height of the bar)
        f'{importance:.4f}',                # Importance value formatted to 4 decimal points
        ha='center', va='bottom', fontsize=10  # Center alignment and font size
    )

plt.xlabel('Features')
plt.ylabel('Mean Permutation Importance')
plt.title('Permutation Importance for KNN Model')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# We attempted to remove payment_of_min_amount due to it's minimal importance as features. However, this did not improve the model's fit.
# 
# **Classification Report**
# 1. Class 1 performs the best with an F1-score of 0.82, demonstrating strong prediction ability for the most frequent class.
# 2. Class 2 performs better compared to the Decision Tree, with an F1-score of 0.76, suggesting the model handles smaller classes reasonably well.
# 3. Class 0 has an F1-score of 0.81, indicating balanced precision and recall.
# 4. Macro average F1-score (0.79): Indicates balanced performance across all classes.
# 5. Weighted average F1-score (0.81): Emphasizes the model's ability to handle the class distribution effectively.
# 
# **Model Suitability**
# 1. KNN handles multi-class problems well when the dataset is scaled, as done here with StandardScaler.
# 2. The model benefits from fewer assumptions but can be computationally intensive for large datasets.
# 
# **Business Implications**
# 1. Insights for Credit Scoring:
#     - Num Credit Card: High importance suggests that the number of credit cards held by an individual directly correlates with their credit score, possibly reflecting their borrowing capacity and financial diversity.
#     - Outstanding Debt: The strong impact of this feature reinforces its role in assessing creditworthiness, critical for risk management strategies.
#     - Interest Rate & Delay from Due Date: These features highlight the importance of financial discipline (e.g., timely repayments and affordable interest rates) in determining credit scores.
# 
# 2. Decision-Making Recommendations:
#     - Focus on Top Features: Target marketing strategies and financial products around top features like num_credit_card and outstanding_debt to identify high-risk and low-risk customers.
# 
# 3. Actionable Insights for Consumers:
#     - Encourage consumers to manage the number of credit cards and outstanding debts responsibly.
#     - Educate consumers about the importance of timely payments and maintaining a healthy credit history.
# 
# 4. Business Strategy:
#     - Leverage insights from KNN to segment customers into actionable categories for loan approvals, credit card offers, or interest rate adjustments.
#     - Provide tailored financial advice based on the key drivers of credit scores identified by the model.

# ### Random Forests

# In[40]:


# from sklearn.ensemble import RandomForestClassifier

# # Define the model
# rf_model = RandomForestClassifier(random_state=42)

# # Define the hyperparameters to tune
# param_grid = {
#     'n_estimators': [50, 100, 200],  # Number of trees in the forest
#     'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
#     'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
#     'min_samples_leaf': [1, 2, 4],   # Minimum samples required to be a leaf node
#     'bootstrap': [True, False]       # Whether to use bootstrap samples
# }

# # Perform grid search with cross-validation
# grid_search = GridSearchCV(
#     estimator=rf_model,
#     param_grid=param_grid,
#     scoring='accuracy',
#     cv=5,
#     verbose=1,
#     n_jobs=-1
# )

# # Fit the model on the training data
# grid_search.fit(X_train, y_train)

# # Get the best model
# best_rf_model = grid_search.best_estimator_

# # Evaluate the model on the validation set
# y_pred = best_rf_model.predict(X_val)

# # Print results
# print("Best Parameters:", grid_search.best_params_)
# print("Training Accuracy:", grid_search.best_score_)
# print("Validation Accuracy:", accuracy_score(y_val, y_pred))
# print("\nClassification Report:\n", classification_report(y_val, y_pred))


# In[41]:


from sklearn.ensemble import RandomForestClassifier

# Define the model with the best parameters
best_rf_model = RandomForestClassifier(
    bootstrap=True,
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=5,
    n_estimators=200,
    random_state=42
)

# Train the model on the training data
best_rf_model.fit(X_train, y_train)

# Predict on the validation set
y_pred = best_rf_model.predict(X_val)

# Calculate and print validation accuracy
val_accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy for Random Forest Classifier: {val_accuracy:.2f}")

# Optionally, print the classification report
print("\nClassification Report:\n", classification_report(y_val, y_pred))


# In[42]:


# Feature Importance Analysis
feature_importances = best_rf_model.feature_importances_

# Replace with your actual feature names
feature_names = [
    'num_bank_accounts', 'num_credit_card', 'num_of_loan', 'credit_mix', 
    'interest_rate', 'credit_history_age', 'delay_from_due_date', 
    'num_credit_inquiries', 'num_of_delayed_payment', 'outstanding_debt'
]

# Create a DataFrame for visualization
importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display the feature importances
# print("\nFeature Importances:\n", importances_df)

# Plot the feature importances with values displayed on the bars
plt.figure(figsize=(10, 6))
bars = plt.bar(importances_df['Feature'], importances_df['Importance'], color='skyblue')

# Annotate each bar with the importance value
for bar, importance in zip(bars, importances_df['Importance']):
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # X position (center of the bar)
        bar.get_height(),                   # Y position (height of the bar)
        f'{importance:.2f}',                # Importance value formatted to 2 decimal points
        ha='center', va='bottom', fontsize=10  # Center alignment and font size
    )

plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances for Random Forest Classifier')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# We attempted to remove num_bank_accounts and payment_of_min_amount due to their minimal importance as features. However, this did not improve the model's fit.
# 
# **Classification Report**
# 1. Class 1 performs the best with an F1-score of 0.83, highlighting the model's ability to handle the majority class effectively.
# 2. Class 2 shows reasonable performance with an F1-score of 0.76, slightly better than Decision Trees but similar to KNN.
# 3. Class 0 performs well with an F1-score of 0.81, indicating balanced precision and recall.
# 4. The macro average F1-score (0.80) reflects balanced performance across all classes.
# 5. The weighted average F1-score (0.81) indicates strong handling of the class distribution.
# 
# **Model Suitability**
# 1. Random Forests are robust to overfitting due to their ensemble nature, making them a reliable choice for this classification problem.
# 2. The model handles both complex interactions and irrelevant features effectively, leveraging its ability to average across multiple trees.
# 
# **Business Implications**
# 1. Insights for Credit Scoring:
#     - Outstanding Debt: As the most significant feature, it highlights the centrality of debt management in determining creditworthiness. High outstanding debt could indicate higher risk, influencing loan and credit card approvals.
#     - Credit History Age: A longer credit history is strongly correlated with better credit scores, reaffirming its importance in credit assessment policies.
#     - Interest Rate & Delay from Due Date: These features suggest the importance of financial discipline and affordable borrowing in influencing credit scores.
# 
# 2. Decision-Making Recommendations:
#     - Risk Management: Focus on top features like outstanding_debt and credit_history_age when assessing customer risk.
#     - Consumer Education: Encourage customers to manage debt effectively and maintain consistent repayment behavior to improve their credit scores.
# 
# 3. Actionable Insights:
#     - Prioritize customers with low outstanding debt and a long credit history for pre-approved financial products.
#     - Monitor customers with high interest_rate or frequent delay_from_due_date for early warning signs of potential defaults.
# 
# 4. Business Strategy: 
# Use the feature importance insights to segment customers into risk groups and offer tailored financial products or interest rates.

# ### Gradient Boosting(CatBoost)

# In[43]:


# !pip install catboost


# In[44]:


# from catboost import CatBoostClassifier

# # Define the model
# catboost_model = CatBoostClassifier(random_state=42, verbose=0)  # Suppress verbose output

# # Define the hyperparameters to tune
# param_grid = {
#     'iterations': [100, 200, 500],       # Number of boosting iterations
#     'learning_rate': [0.01, 0.05, 0.1],  # Learning rate
#     'depth': [4, 6, 8],                 # Depth of the tree
#     'l2_leaf_reg': [1, 3, 5],           # Regularization parameter
#     'border_count': [32, 64, 128]       # Number of splits for numerical features
# }

# # Perform grid search with cross-validation
# grid_search = GridSearchCV(
#     estimator=catboost_model,
#     param_grid=param_grid,
#     scoring='accuracy',
#     cv=5,
#     verbose=1,
#     n_jobs=-1
# )

# # Fit the model on the training data
# grid_search.fit(X_train, y_train)

# # Get the best model
# best_catboost_model = grid_search.best_estimator_

# # Evaluate the model on the validation set
# y_pred = best_catboost_model.predict(X_val)

# # Print results
# print("Best Parameters:", grid_search.best_params_)
# print("Training Accuracy:", grid_search.best_score_)
# print("Validation Accuracy:", accuracy_score(y_val, y_pred))
# print("\nClassification Report:\n", classification_report(y_val, y_pred))


# In[45]:


from catboost import CatBoostClassifier

# Define the model with the best parameters
best_catboost_model = CatBoostClassifier(
    border_count=64,
    depth=8,
    iterations=500,
    l2_leaf_reg=1,
    learning_rate=0.1,
    random_state=42,
    verbose=0  # Suppress verbose output
)

# Train the model on the training data
best_catboost_model.fit(X_train, y_train)

# Predict on the validation set
y_pred = best_catboost_model.predict(X_val)

# Calculate and print validation accuracy
val_accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy for CatBoost Classifier: {val_accuracy:.2f}")

# Optionally, print the classification report
print("\nClassification Report:\n", classification_report(y_val, y_pred))


# In[46]:


# Feature Importance Analysis
feature_importances = best_catboost_model.get_feature_importance()

# Replace with your actual feature names
feature_names = [
    'num_bank_accounts', 'num_credit_card', 'num_of_loan', 'credit_mix', 
    'interest_rate', 'credit_history_age', 'delay_from_due_date', 
    'num_credit_inquiries', 'num_of_delayed_payment', 'outstanding_debt'
]

# Create a DataFrame for visualization
importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display the feature importances
# print("\nFeature Importances:\n", importances_df)

# Plot the feature importances with values displayed on the bars
plt.figure(figsize=(10, 6))
bars = plt.bar(importances_df['Feature'], importances_df['Importance'], color='skyblue')

# Annotate each bar with the importance value
for bar, importance in zip(bars, importances_df['Importance']):
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # X position (center of the bar)
        bar.get_height(),                   # Y position (height of the bar)
        f'{importance:.2f}',                # Importance value formatted to 2 decimal points
        ha='center', va='bottom', fontsize=10  # Center alignment and font size
    )

plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances for CatBoost Classifier')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# We attempted to remove payment_of_min_amount due to it's minimal importance as feature. However, this did not improve the model's fit.
# 
# **Classification Report**
# 1. Class 1 performs the best with an F1-score of 0.80, demonstrating robust predictions for the majority class.
# 2. Class 2 has an F1-score of 0.71, indicating that the model struggles with smaller classes.
# 3. Class 0 achieves an F1-score of 0.76, showing balanced precision and recall.
# 4. Macro average F1-score (0.76): Indicates a slight imbalance in class performance.
# 5. Weighted average F1-score (0.78): Reflects the model's focus on the majority class while still accounting for the minority classes.
# 
# **Model Suitability**
# 1. CatBoost handles categorical and numerical features efficiently, offering robust performance without extensive preprocessing.
# 2. The model's slightly lower accuracy compared to Random Forest may result from its sensitivity to hyperparameter tuning or class imbalance.
# 
# **Business Implications**
# 1. Insights for Credit Scoring:
#     - Outstanding Debt: As the most important feature, it underscores the need for financial institutions to monitor debt levels closely for creditworthiness assessments.
#     - Interest Rate: Its high importance reflects the impact of borrowing costs on credit scores, making it a critical factor in loan approvals.
#     - Credit History Age: The significance of this feature highlights the need for long-term credit behavior in evaluating financial reliability.
#     - Delay from Due Date: This feature emphasizes the importance of timely repayments in maintaining a good credit score.
# 
# 2. Decision-Making Recommendations:
#     - Targeting High-Risk Customers: Use insights from top features to identify and monitor high-risk customers for preemptive risk mitigation.
#     - Product Customization: Offer lower interest rates or personalized loan terms to customers with strong credit histories or consistent repayment behavior.
#     - Educating Customers: Raise awareness about the importance of managing debt levels and paying dues on time to improve credit scores.
# 
# 3. Business Strategy:
#     - Segment customers based on their credit risk using the top features identified.
#     - Provide incentives, such as better loan terms, to customers with low outstanding debt or good repayment histories.

# ### Neural Network(MLP Classifier)

# In[47]:


# from sklearn.neural_network import MLPClassifier

# # Define the model
# mlp_model = MLPClassifier(random_state=42, max_iter=1000)

# # Define the hyperparameters to tune
# param_grid = {
#     'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],  # Number of neurons in hidden layers
#     'activation': ['relu', 'tanh', 'logistic'],                 # Activation function
#     'solver': ['adam', 'sgd'],                                  # Optimization algorithm
#     'alpha': [0.0001, 0.001, 0.01],                             # L2 regularization parameter
#     'learning_rate': ['constant', 'adaptive']                   # Learning rate schedule
# }

# # Perform grid search with cross-validation
# grid_search = GridSearchCV(
#     estimator=mlp_model,
#     param_grid=param_grid,
#     scoring='accuracy',
#     cv=5,
#     verbose=1,
#     n_jobs=-1
# )

# # Fit the model on the training data
# grid_search.fit(X_train, y_train)

# # Get the best model
# best_mlp_model = grid_search.best_estimator_

# # Evaluate the model on the validation set
# y_pred = best_mlp_model.predict(X_val)

# # Print results
# print("Best Parameters:", grid_search.best_params_)
# print("Training Accuracy:", grid_search.best_score_)
# print("Validation Accuracy:", accuracy_score(y_val, y_pred))
# print("\nClassification Report:\n", classification_report(y_val, y_pred))


# In[48]:


from sklearn.neural_network import MLPClassifier

# Define the model with the best parameters
best_mlp_model = MLPClassifier(
    activation='tanh',
    alpha=0.001,
    hidden_layer_sizes=(100, 50),
    learning_rate='constant',
    solver='adam',
    random_state=42,
    max_iter=2500  # Ensure convergence
)

# Train the model on the training data
best_mlp_model.fit(X_train, y_train)

# Predict on the validation set
y_pred = best_mlp_model.predict(X_val)

# Calculate and print validation accuracy
val_accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy for Neural Network(MLP Classifier): {val_accuracy:.2f}")

# Optionally, print the classification report
print("\nClassification Report:\n", classification_report(y_val, y_pred))


# In[49]:


import warnings
warnings.filterwarnings('ignore')


# In[50]:


# !pip install torch torchvision torchaudio


# In[51]:


# !pip install captum


# In[52]:


import torch
from captum.attr import IntegratedGradients

# Convert the MLP model to a PyTorch model
class TorchMLP(torch.nn.Module):
    def __init__(self, sk_model, input_dim):
        super(TorchMLP, self).__init__()
        layers = []
        for layer_size in sk_model.hidden_layer_sizes:
            layers.append(torch.nn.Linear(input_dim, layer_size))
            layers.append(torch.nn.Tanh())  # Match the activation function
            input_dim = layer_size
        layers.append(torch.nn.Linear(input_dim, sk_model.n_outputs_))  # Output layer
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Initialize the TorchMLP with the correct input dimension
input_dim = X_train.shape[1]  # Number of features in X_train
torch_model = TorchMLP(best_mlp_model, input_dim)

# Define the Integrated Gradients object
ig = IntegratedGradients(torch_model)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)  # Convert NumPy array to PyTorch tensor

# Compute attributions using Integrated Gradients
attr, delta = ig.attribute(X_val_tensor, target=0, return_convergence_delta=True)

# Aggregate the feature attributions (average across the validation set)
attr_mean = attr.detach().numpy().mean(axis=0)

# Dynamically set feature names
feature_names = [
    'num_bank_accounts', 'num_credit_card', 'num_of_loan', 'credit_mix', 
    'interest_rate', 'credit_history_age', 'delay_from_due_date', 
    'num_credit_inquiries', 'num_of_delayed_payment', 'outstanding_debt'
]

# Ensure no mismatch
if len(feature_names) != len(attr_mean):
    raise ValueError("Mismatch between number of features and attributions.")

# Create DataFrame for visualization
importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': attr_mean
}).sort_values(by='Importance', ascending=False)

# Display the feature importances
# print("\nIntegrated Gradients Feature Importances:\n", importances_df)

# Plot the feature importances
plt.figure(figsize=(10, 6))
bars = plt.bar(importances_df['Feature'], importances_df['Importance'], color='skyblue')

# Annotate each bar with the importance value
for bar, importance in zip(bars, importances_df['Importance']):
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # X position (center of the bar)
        bar.get_height(),                   # Y position (height of the bar)
        f'{importance:.4f}',                # Importance value formatted to 4 decimal points
        ha='center', va='bottom', fontsize=10  # Center alignment and font size
    )

plt.xlabel('Features')
plt.ylabel('Integrated Gradients Importance')
plt.title('Feature Importances for Neural Network (Integrated Gradients)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# ### Predictions for Test File

# In[53]:


test = pd.read_csv('credit_score_test.csv')


# In[54]:


test.columns = map(str.lower, test.columns)


# In[55]:


test.drop(['id', 'customer_id', 'month', 'name', 'ssn', 'occupation', 'type_of_loan', 'payment_behaviour'], axis=1, inplace=True)


# In[56]:


credit_mix_map = {'Bad': 0, 'Standard': 1, 'Good': 2}
test['credit_mix'] = test['credit_mix'].map(credit_mix_map)


# In[57]:


min_amount_map = {'No': 0, 'Yes': 1}
test['payment_of_min_amount'] = test['payment_of_min_amount'].map(min_amount_map)


# In[58]:


# Define bins and labels for num_bank_accounts
bins_bank_accounts = [0, 2, 5, 8, float('inf')]
labels_bank_accounts = [0, 1, 2, 3]
test['num_bank_accounts'] = pd.cut(test['num_bank_accounts'], bins=bins_bank_accounts, labels=labels_bank_accounts, right=True).astype(float)


# In[59]:


import re

# Remove special characters from the 'num_of_loan' column
test['num_of_loan'] = test['num_of_loan'].apply(lambda x: re.sub('[^0-9]', '', str(x)))

# Convert the 'num_of_loan' column to numeric
test['num_of_loan'] = pd.to_numeric(test['num_of_loan'], errors='coerce')


# In[60]:


# None => 0
# Very Low => 1
# Low => 2
# Moderate => 3
# High => 4

# Define bins and labels for num_of_loan
bins_num_of_loan = [-1, 0, 2, 4, 6, float('inf')]
labels_num_of_loan = [0, 1, 2, 3, 4]

# Create a new binned column for num_of_loan
test['num_of_loan'] = pd.cut(test['num_of_loan'], bins=bins_num_of_loan, labels=labels_num_of_loan, right=True).astype(float)


# In[61]:


# None => 0
# Very Low => 1
# Low => 2
# Moderate => 3
# High => 4

# Define bins and labels for num_credit_card
bins_num_credit_card = [-1, 0, 2, 4, 7, float('inf')]
labels_num_credit_card = [0, 1, 2, 3, 4]

# Create a new binned column for num_credit_card
test['num_credit_card'] = pd.cut(test['num_credit_card'], bins=bins_num_credit_card, labels=labels_num_credit_card, right=True).astype(float)


# In[62]:


test.dropna(inplace=True)


# In[63]:


# Convert the specified columns to integer type
columns_to_convert = ['num_bank_accounts', 'num_credit_card', 'num_of_loan', 'payment_of_min_amount', 'credit_mix']

for col in columns_to_convert:
    test[col] = test[col].astype(int)


# In[64]:


# Function to convert 'Years and Months' format to total months
def convert_to_months(value):
    # Use regular expressions to extract years and months
    years_match = re.search(r'(\d+)\s*Years?', str(value))
    months_match = re.search(r'(\d+)\s*Months?', str(value))
    
    # Extract integer values (default to 0 if no match found)
    years = int(years_match.group(1)) if years_match else 0
    months = int(months_match.group(1)) if months_match else 0
    
    # Convert to total months
    return (years * 12) + months

# Apply the function to the 'credit_history_age' column
test['credit_history_age'] = test['credit_history_age'].apply(convert_to_months)


# In[65]:


# Function to extract numeric values and convert to appropriate dtype
def extract_numeric(df, column_name):
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')  # Converts non-numeric to NaN
    return df

# Apply function to the required columns
test = extract_numeric(test, 'num_of_delayed_payment')
test = extract_numeric(test, 'outstanding_debt')


# In[66]:


test.dropna(inplace=True)


# In[67]:


test


# In[68]:


# Ensure the test dataset has only the required features for prediction
selected_features = [
    'num_bank_accounts', 'num_credit_card', 'num_of_loan', 'credit_mix', 
    'interest_rate', 'credit_history_age', 'delay_from_due_date', 
    'num_credit_inquiries', 'num_of_delayed_payment', 'outstanding_debt'
]

# Subset the test dataset with selected features
X_test = test[selected_features]

# Predict credit_score using the trained Random Forest model
y_pred = best_rf_model.predict(X_test)

# Store the predictions in a new column 'credit_score'
test['credit_score'] = y_pred

# Save the results to a new CSV file
output_file = 'credit_score_results.csv'
test.to_csv(output_file, index=False)

print(f"Predictions saved to {output_file}")


# In[69]:


test

