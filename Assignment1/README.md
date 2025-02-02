# **Project Report: Fashion MNIST Classification with Dimensionality Reduction**

## **1. Suitable Name for the Project**
**Fashion MNIST Classifier with Feature Reduction**

## **2. Project Title**
**Evaluating Dimensionality Reduction for Fashion MNIST Classification Using Random Forest and KNN**

## **3. Project Description**
This project aims to classify images from the Fashion MNIST dataset using machine learning models while investigating the impact of dimensionality reduction techniques. The study evaluates Kernel PCA, Locally Linear Embedding (LLE), and Isomap in combination with Random Forest and K-Nearest Neighbors (KNN) classifiers. The objective is to identify the best-performing approach and determine whether dimensionality reduction enhances classification accuracy.

## **4. Overview Description with Key Steps**
The workflow of this project consists of the following steps:

1. **Data Loading & Preprocessing**  
   - Load the Fashion MNIST dataset and remove duplicates.
   - Extract labels and features, using only the first 1500 samples for consistency.

2. **Train-Test Split & Standardization**  
   - Split data into training and testing sets using stratified sampling.
   - Apply feature scaling using `StandardScaler`.

3. **Dimensionality Reduction Techniques**  
   - Implement Kernel PCA, Locally Linear Embedding (LLE), and Isomap with 15 components.

4. **Model Training & Hyperparameter Tuning**  
   - Train Random Forest and KNN models with each dimensionality reduction technique.
   - Perform hyperparameter tuning using Grid Search with 5-fold Stratified K-Fold cross-validation.

5. **Evaluation & Comparison**  
   - Evaluate accuracy on the test set.
   - Compare results with and without dimensionality reduction to determine the best-performing approach.

## **5. Goals**
### **Primary Goal:**
- Develop an effective classification model for Fashion MNIST that maximizes accuracy.

### **Secondary Goals:**
1. **Evaluate Dimensionality Reduction Impact:**  
   - Assess whether dimensionality reduction improves classification performance or leads to information loss.

2. **Hyperparameter Optimization:**  
   - Identify the best hyperparameter settings for Random Forest and KNN.

3. **Comparison of Model Performance:**  
   - Determine which classifier (Random Forest vs. KNN) is more effective for this dataset.

## **6. Results**
### **Summary of Findings**
- **Kernel PCA with Random Forest** achieved the highest accuracy among the dimensionality reduction methods (~75.53%).
- **Using the original dataset without dimensionality reduction** yielded the best overall accuracy of **87.71%** with Random Forest.
- Dimensionality reduction techniques led to a loss of useful information, negatively impacting classification performance.

## **7. Results Pointers (2-3 Key Findings)**
1. **Best Model Without Dimensionality Reduction:**  
   - Random Forest without feature reduction achieved the highest accuracy (87.71%), indicating that reducing dimensions might discard essential patterns in the data.

2. **Kernel PCA Performed Best Among Feature Reduction Methods:**  
   - Kernel PCA outperformed LLE and Isomap when combined with both Random Forest and KNN, likely due to its ability to capture non-linear relationships.

3. **Random Forest Outperformed KNN in All Cases:**  
   - Across all experiments, Random Forest consistently outperformed KNN, highlighting the effectiveness of ensemble learning for this classification task.

## **8. Tech Stack & Algorithms Used**
### **Libraries & Tools:**
- **Pandas** for data manipulation.
- **Scikit-Learn** for machine learning models, preprocessing, dimensionality reduction, and hyperparameter tuning.

### **Machine Learning Models:**
- **Random Forest Classifier** (ensemble learning method)
- **K-Nearest Neighbors (KNN)** (instance-based learning)

### **Dimensionality Reduction Techniques:**
- **Kernel PCA** (Non-linear feature extraction)
- **Locally Linear Embedding (LLE)** (Manifold learning)
- **Isomap** (Geodesic distance-based reduction)

### **Optimization Methods:**
- **Grid Search with Cross-Validation** (Hyperparameter tuning)
- **Stratified K-Fold (5 splits)** (Ensuring balanced class distribution)
