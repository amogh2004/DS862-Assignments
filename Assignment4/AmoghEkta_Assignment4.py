#!/usr/bin/env python
# coding: utf-8

# # ASSIGNMENT-4

# # Group-Name:Amogh Ranganathaih; Name:Ekta Singh

# In[1]:


# For this assignment, you will be trying out different structures of MLP and compare the performance. We will again  work on a regression data set and a classification data set.


# In[2]:


import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


# # PART 1: Regression

# In[3]:


housing = fetch_california_housing()
X = housing.data
y = housing.target


# In[4]:


# We have 8 features and 20640 observations.
X.shape


# In[5]:


# Define column names for X
X_columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

# Convert X to a DataFrame with headers
X_df = pd.DataFrame(X, columns=X_columns)

# Convert y to a DataFrame and add the column name 'MedValue'
y_df = pd.DataFrame(y, columns=['MedValue'])

# Combine X and y into a single DataFrame
housing_df = pd.concat([X_df, y_df], axis=1)


# In[6]:


# Display the DataFrame
housing_df


# # TASK 1

# In[7]:


# Split the data into training, validation, and testing sets. Scale the training set and apply the same scale onto the validation and testing sets. Make sure you set random seed to the result is reproducible.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split the data into training (60%), validation (20%), and testing sets (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize the scaler
scaler = StandardScaler()

# Scale the training data
X_train_scaled = scaler.fit_transform(X_train)

# Same scaling to the validation and test data
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Optional: Check the shape of the datasets
print(f'Training set shape: {X_train_scaled.shape}')
print(f'Validation set shape: {X_val_scaled.shape}')
print(f'Test set shape: {X_test_scaled.shape}')


# # TASK 2

# In[8]:


get_ipython().system('pip install tensorflow')


# In[9]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

# Define the model
model = Sequential()

# Add the input layer explicitly
model.add(Input(shape=(X_train_scaled.shape[1],)))

# Add the first hidden layer with 15 neurons
model.add(Dense(15, activation='relu'))

# Add the second hidden layer with 10 neurons
model.add(Dense(10, activation='relu'))

# Add the output layer with 1 neuron (for regression tasks)
model.add(Dense(1))

# Compile the model with mean squared error loss and Adam optimizer
model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping callback configuration
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=50,         # Stop after 10 epochs without improvement
    restore_best_weights=True  # Restore the model weights from the epoch with the best validation loss
)
# Fit the model to the training data with early stopping
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,               # Maximum number of epochs
    batch_size=32,
    validation_data=(X_val_scaled, y_val),
    callbacks=[early_stopping],  # Early stopping
    verbose=0
)


# In[10]:


# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Calculate the Mean Squared Error on the test set
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on the test set: {mse}')


# # TASK 3

# In[11]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

# Define the model
model = Sequential()

# Add the input layer explicitly
model.add(Input(shape=(X_train_scaled.shape[1],)))

# Add the hidden layers with the specified structure
model.add(Dense(7, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(2, activation='relu'))

# Add the output layer with 1 neuron (for regression tasks)
model.add(Dense(1))

# Compile the model with mean squared error loss and Adam optimizer
model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping callback configuration
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=20,         # Stop after 10 epochs without improvement
    restore_best_weights=True  # Restore the model weights from the epoch with the best validation loss
)

# Fit the model to the training data with early stopping
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,               # Maximum number of epochs
    batch_size=32,
    validation_data=(X_val_scaled, y_val),
    callbacks=[early_stopping],  # Early stopping
    verbose=0
)


# In[12]:


# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Calculate the Mean Squared Error on the test set
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on the test set: {mse}')


# In[13]:


#The shallow neural network with two hidden layers of 15 and 10 neurons achieved a Mean Squared Error (MSE) of 0.286 on the test set, indicating a reasonable performance for the regression task. In contrast, the deeper network with five hidden layers consisting of 7, 5, 3, 2, and 2 neurons resulted in a significantly higher MSE of 1.373. This substantial increase in error suggests that the deeper architecture may have led to overfitting, despite having fewer neurons per layer. Consequently, the shallow network is preferred in this case, as it demonstrates better generalization on the test data, indicating it captures the underlying patterns more effectively without overfitting.


# # TASK 4

# In[14]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Gradient Boosting with customized parameters
gbr_custom = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)

# Fit the model
gbr_custom.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_gbr_custom = gbr_custom.predict(X_test_scaled)

# Calculate the MSE
mse_gbr_custom = mean_squared_error(y_test, y_pred_gbr_custom)
print(f'Mean Squared Error (Tuned Gradient Boosting) on the test set: {mse_gbr_custom}')

# Calculate the R-squared value
r2_gbr_custom = r2_score(y_test, y_pred_gbr_custom)
print(f'R-squared (Tuned Gradient Boosting) on the test set: {r2_gbr_custom}')


# In[15]:


#The Gradient Boosting Regressor with customized parameters outperforms both MLP models. It achieved a significantly lower Mean Squared Error (MSE) of 0.2451 compared to the MLP model's MSE of 0.285 and 1.37. Additionally, it achieved an R-squared value of 0.8213, indicating better explanatory power and model fit compared to the MLP models, which do not provide an R-squared measure. This suggests that for this regression task, the Gradient Boosting Regressor is more effective at capturing patterns in the data and provides a better overall performance compared to the neural network-based approaches.


# # PART 2: Classification

# In[16]:


mobile = pd.read_csv('/Users/vega/Desktop/mobile.csv')


# In[17]:


y = mobile.price_range
del mobile['price_range']
X = mobile


# # TASK 1

# In[18]:


# Split the data into training (60%), validation (20%), and testing sets (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize the scaler
scaler = StandardScaler()

# Scale the training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same scaling to the validation and test data
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Optional: Check the shape of the datasets
print(f'Training set shape: {X_train_scaled.shape}')
print(f'Validation set shape: {X_val_scaled.shape}')
print(f'Test set shape: {X_test_scaled.shape}')


# # TASK 2

# In[19]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error

# Initialize the neural network model
model = Sequential()

# Input layer with the number of features as input_dim
input_dim = X_train_scaled.shape[1]

# Add the first hidden layer with 64 neurons and ReLU activation
model.add(Dense(64, input_dim=input_dim, activation='relu'))

# Add the second hidden layer with 32 neurons and ReLU activation
model.add(Dense(32, activation='relu'))

# Add the third hidden layer with 16 neurons and ReLU activation
model.add(Dense(16, activation='relu'))

# Add the output layer with 1 neuron and softmax activation (for classification)
model.add(Dense(4, activation='softmax'))  # Assuming 4 classes for 'price_range'

# Compile the model with categorical crossentropy as loss (for multi-class classification)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), epochs=100, batch_size=32, verbose=1)



# In[20]:


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'Test Accuracy: {test_accuracy:.4f}')

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate Mean Squared Error (MSE) on the test set
mse = mean_squared_error(y_test, y_pred.argmax(axis=1))
print(f'Test MSE: {mse:.4f}')


# # TASK 3

# In[21]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

# Gradient Boosting Classifier with customized parameters
gbc_custom = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)

# Fit the model
gbc_custom.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_gbc_custom = gbc_custom.predict(X_test_scaled)

# Calculate the accuracy
accuracy_gbc_custom = accuracy_score(y_test, y_pred_gbc_custom)
print(f'Accuracy (Tuned Gradient Boosting Classifier) on the test set: {accuracy_gbc_custom:.4f}')

# Calculate the Mean Squared Error (MSE) for comparison purposes
mse_gbc_custom = mean_squared_error(y_test, y_pred_gbc_custom)
print(f'Mean Squared Error (Tuned Gradient Boosting Classifier) on the test set: {mse_gbc_custom:.4f}')


# In[22]:


#Both the Neural Network and the Gradient Boosting Classifier (GBC) perform very well on the classification task for predicting mobile price range, with the GBC slightly outperforming the Neural Network. The Gradient Boosting Classifier achieved a marginally higher accuracy of 97.25% compared to the Neural Network’s 96.50%. Additionally, the Mean Squared Error (MSE) for GBC is lower (0.0275) than that of the Neural Network (0.0350), indicating a slight edge in prediction accuracy. Although both models yield excellent results, the Gradient Boosting Classifier shows slightly better performance, making it the better method in this case.


# In[ ]:





# In[ ]:





# In[ ]:




