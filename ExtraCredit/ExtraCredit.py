#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# In[2]:


# Set paths
train_dir = './train'  # Train folder in the same directory as the notebook
img_size = 64  # Resize all images to 64x64

# Load and preprocess data
def load_data(data_dir):
    X, y = [], []
    classes = sorted(os.listdir(data_dir))
    class_map = {cls: idx for idx, cls in enumerate(classes)}  # Map class names to indices
    
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue  # Skip if it's not a directory
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            img = cv2.imread(img_path)  # Read the image in color (default is BGR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img = cv2.resize(img, (img_size, img_size))  # Resize to uniform size
                X.append(img)
                y.append(class_map[cls])  # Append the corresponding class label
    return np.array(X), np.array(y), class_map


# In[3]:


# Load data
X, y, class_map = load_data(train_dir)

# Normalize pixel values (scale to range 0-1)
X = X / 255.0

# Convert labels to one-hot encoded format
y = to_categorical(y)

# Split into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Summary
print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_test.shape[0]}")
print(f"Image shape: {X_train.shape[1:]}")
print(f"Number of classes: {len(class_map)}")


# In[4]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=15,         # Random rotation between -15 to +15 degrees
    width_shift_range=0.1,     # Random horizontal shift up to 10% of image width
    height_shift_range=0.1,    # Random vertical shift up to 10% of image height
    zoom_range=0.1,            # Random zoom in/out by 10%
    horizontal_flip=True,      # Randomly flip images horizontally
    fill_mode='nearest'        # Fill any empty pixels after transformation
)

# Fit the data augmentation generator to the training data
datagen.fit(X_train)


# In[5]:


import matplotlib.pyplot as plt

# Display a few augmented images
def visualize_augmentation_fixed(generator, images, labels, class_map, num_images=5):
    for X_batch, y_batch in generator.flow(images, labels, batch_size=num_images):
        plt.figure(figsize=(10, 10))
        for i in range(0, num_images):
            plt.subplot(1, num_images, i + 1)
            # Rescale images to 0-255 for proper visualization
            img = (X_batch[i] * 255).astype('uint8')
            plt.imshow(img)  # No conversion needed since the image is in RGB
            class_label = list(class_map.keys())[np.argmax(y_batch[i])]
            plt.title(class_label)
            plt.axis('off')
        break  # Only visualize one batch

# Visualize augmented training samples
visualize_augmentation_fixed(datagen, X_train, y_train, class_map)


# In[6]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def build_cnn_with_lower_dropout(input_shape, num_classes):
    model = Sequential()

    # Convolutional Block 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))  # Reduced from 0.2 to 0.1

    # Convolutional Block 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))  # Reduced from 0.2 to 0.1

    # Flattening Layer
    model.add(Flatten())

    # Fully Connected Layer
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))  # Reduced from 0.4 to 0.2

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))  # Output layer for classification

    return model

# Model Parameters
input_shape = (img_size, img_size, 3)  # Color images
num_classes = len(class_map)

# Build and compile the model with reduced dropout rates
reduced_dropout_model = build_cnn_with_lower_dropout(input_shape, num_classes)

reduced_dropout_model.compile(optimizer='adam', 
                              loss='categorical_crossentropy', 
                              metrics=['accuracy'])

# Summary of the model
reduced_dropout_model.summary()


# In[7]:


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Callbacks for training
early_stopping = EarlyStopping(
    monitor='val_loss',       # Stop when validation loss stops improving
    patience=5,               # Wait for 5 epochs before stopping
    restore_best_weights=True # Restore the best model weights
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',       # Monitor validation loss
    factor=0.5,               # Reduce learning rate by half
    patience=3,               # After 3 epochs of no improvement
    min_lr=1e-6               # Minimum learning rate
)

# Batch size and epochs
batch_size = 32
epochs = 50

# Train the model
history = reduced_dropout_model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),  # Augmented training data
    validation_data=(X_test, y_test),                        # Validation data
    epochs=epochs,                                         # Maximum number of epochs
    callbacks=[early_stopping, reduce_lr],                 # Early stopping and learning rate reduction
    verbose=1                                              # Print training progress
)


# In[8]:


import matplotlib.pyplot as plt

# Plot training and validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[9]:


# Evaluate the model on the test dataset
test_loss, test_accuracy = reduced_dropout_model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")


# In[10]:


import pandas as pd
import os
import numpy as np

# Load the CSV file
csv_file = 'test.csv'  # Replace with the correct name if necessary
test_df = pd.read_csv(csv_file)

# Ensure the CSV file has the column for image filenames and the label column
print(test_df.head())  # Check the structure of the CSV

# Preprocess test images
def preprocess_test_images_from_csv(test_df, img_size, test_dir):
    test_images = []
    image_names = []
    
    for img_name in test_df['File']:  # Assuming 'filename' is the column name for image files
        img_path = os.path.join(test_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = cv2.resize(img, (img_size, img_size)) / 255.0  # Normalize
            test_images.append(img)
            image_names.append(img_name)
    
    return np.array(test_images), image_names

# Path to the test images directory
test_dir = './test'  # Update if needed

# Preprocess test images
X_test, test_image_names = preprocess_test_images_from_csv(test_df, img_size, test_dir)

# Generate predictions
predictions = reduced_dropout_model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Map predicted classes back to labels
class_labels = {v: k for k, v in class_map.items()}  # Reverse mapping of class_map
predicted_labels = [class_labels[cls] for cls in predicted_classes]

# Update the CSV with predictions
test_df['label'] = predicted_labels

# Save the updated CSV file
output_csv_file = 'test_with_predictions.csv'
test_df.to_csv(output_csv_file, index=False)

print(f"Updated CSV with predictions saved to {output_csv_file}")

