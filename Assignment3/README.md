### **Project Report: Customer Churn Prediction Using SVM & Logistic Regression**

## **1. Suitable Name for the Project**
**ChurnGuard: Predicting Customer Attrition with SVM and Logistic Regression**

## **2. Project Title**
**Machine Learning for Customer Churn Prediction: A Comparative Study of SVM and Logistic Regression**

## **3. Project Description**
This project aims to predict customer churn using a dataset obtained from Kaggle. The dataset contains various customer attributes, such as service type, tenure, and contract type. The goal is to build a predictive model that can classify customers as "churn" or "not churn."

The project employs **Support Vector Machines (SVM) with Linear, Polynomial, and RBF Kernels** and **Regularized Logistic Regression** to compare their effectiveness in predicting customer churn.

---

## **4. Overview Description with Key Steps**
The project workflow consists of:

1. **Data Preprocessing:**
   - Load the dataset and remove irrelevant features (`customerID`, etc.).
   - Encode categorical variables using one-hot encoding.
   - Convert the target variable (`Churn`) into numerical format.

2. **Train-Test Split:**
   - Split the dataset into **80% training and 20% testing**.

3. **SVM Model Training & Evaluation:**
   - Train **Linear Kernel SVM** and evaluate classification error.
   - Use **GridSearchCV** to tune **Polynomial Kernel SVM** and **RBF Kernel SVM**.

4. **Regularized Logistic Regression:**
   - Train a logistic regression model and compare its performance against SVM models.

5. **Model Evaluation:**
   - Use **Accuracy, Precision, Recall, F1-score, Confusion Matrices**, and **ROC Curves**.
   - Determine which model provides the best classification performance.

---

## **5. Goals**
### **Primary Goal**
- Build an effective churn prediction model that accurately identifies customers likely to leave.

### **Secondary Goals**
1. **Evaluate SVM Performance:**  
   - Compare different **SVM kernels (Linear, Polynomial, RBF)** to determine the best-performing model.

2. **Compare Against Logistic Regression:**  
   - Investigate whether **regularized logistic regression** can outperform SVM models in churn prediction.

3. **Optimize Model Hyperparameters:**  
   - Use **GridSearchCV** to tune **C, gamma, and kernel parameters** for the best model performance.

---

## **6. Results**
### **Summary of Findings**
- **Polynomial Kernel SVM achieved the highest AUC (0.83)**, indicating the best separation between churn and non-churn customers.
- **RBF Kernel SVM performed closely (AUC = 0.82)** but had slightly lower recall.
- **Linear Kernel SVM was the weakest performer (AUC = 0.69)**, struggling to capture non-linear relationships.
- **Logistic Regression performed slightly worse than Polynomial SVM** but was computationally efficient.
- **Feature selection and hyperparameter tuning significantly improved model performance.**

---

## **7. Results Pointers (2-3 Key Findings)**
1. **Polynomial Kernel SVM Outperformed Other Models:**  
   - Achieved the highest **recall and AUC (0.83)**, making it the best choice for predicting customer churn.

2. **Linear Kernel SVM Struggled to Capture Complex Relationships:**  
   - AUC was **only 0.69**, indicating that a linear decision boundary is insufficient for this problem.

3. **Hyperparameter Tuning Significantly Improved SVM Performance:**  
   - **GridSearchCV optimization of C, degree (Polynomial), and gamma (RBF)** led to better classification accuracy.

---

## **8. Tech Stack & Algorithms Used**
### **Libraries & Tools**
- **Pandas, NumPy** - Data processing
- **Scikit-learn** - Model training, hyperparameter tuning, and evaluation
- **Matplotlib, Seaborn** - Data visualization

### **Machine Learning Models**
- **Support Vector Machine (SVM)**
  - **Linear Kernel**
  - **Polynomial Kernel**
  - **RBF Kernel**
- **Regularized Logistic Regression**

### **Evaluation Metrics**
- **Accuracy, Precision, Recall, F1-score**
- **Confusion Matrix**
- **ROC Curve & AUC Score**
- **Hyperparameter Optimization (GridSearchCV, RandomizedSearchCV)**

---

### **Final Conclusion**
- **Polynomial Kernel SVM is the best-performing model for churn prediction.**
- **Logistic Regression provides a simpler, more interpretable alternative with comparable performance.**
- **Feature selection and hyperparameter tuning significantly impact model accuracy and recall.**
- **Further improvements can be made by testing ensemble methods (Boosting, Bagging) or incorporating domain-specific features.**
![image](https://github.com/user-attachments/assets/42be4b09-6b2a-4b6b-a48e-2865b376c3f5)
