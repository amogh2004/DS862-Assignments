### **Project Report: Ensemble Learning for Bank Churn Prediction**

## **1. Suitable Name for the Project**
**Ensemble Learning for Predicting Bank Customer Churn**

## **2. Project Title**
**Comparing Ensemble Learning Techniques for Bank Customer Churn Prediction**

## **3. Project Description**
This project applies **ensemble learning techniques** to predict whether a bank customer will churn based on various financial and demographic factors. Using the **Bank Churn dataset** from Kaggle, multiple ensemble models were trained, tuned, and evaluated to determine the most effective classification method.  

We implemented and compared the following ensemble methods:
1. **Voting Classifier (Hard & Soft Voting)**
2. **Bagging (SVM as base learner)**
3. **Boosting (XGBoost & LightGBM)**
4. **Stacking (Blending predictions of multiple models)**
5. **Random Forest, AdaBoost, and Gradient Boosting classifiers**

The goal was to compare their performance in terms of **accuracy on the test set** and determine the most effective ensemble strategy for churn prediction.

---

## **4. Overview Description with Key Steps**
The project workflow consists of the following steps:

### **Step 1: Data Preprocessing**
- Load the dataset and **drop unnecessary columns** (`RowNumber`, `CustomerId`, `Surname`).
- Convert categorical variables into **dummy variables**.
- **Standardize numerical features** using **StandardScaler**.
- Split the dataset into **80% training and 20% test data**.

---

### **Step 2: Model Training and Evaluation**
1. **Hyperparameter Tuning for Individual Models**
   - Tune hyperparameters for **KNN, Decision Trees, Random Forest, and SVM** using **GridSearchCV**.

2. **Voting Classifier**
   - **Soft Voting** (Weighted probability averaging of classifiers).
   - **Hard Voting** (Majority rule across classifiers).
   - Compare their accuracy on the test set.

3. **Bagging (Bootstrap Aggregation)**
   - Use **BaggingClassifier with SVM as base learner**.
   - Evaluate Out-of-Bag (OOB) error.

4. **Boosting**
   - Train **XGBoost** and **LightGBM** classifiers.
   - Compare performance against bagging models.

5. **Stacking**
   - Base models: **RandomForest, AdaBoost, GradientBoosting, SVC**.
   - Meta-learner: **Logistic Regression**.

---

## **5. Goals**
### **Primary Goal**
- Use **ensemble learning** to improve customer churn prediction accuracy.

### **Secondary Goals**
1. **Evaluate the effectiveness of different ensemble methods.**
   - Compare **Voting, Bagging, Boosting, and Stacking**.

2. **Tune Hyperparameters for Best Performance.**
   - Optimize individual models before combining them.

3. **Compare Accuracy Between Base Learners and Ensemble Models.**
   - Identify whether **ensemble models outperform individual classifiers**.

---

## **6. Results**
### **Performance Comparison of Ensemble Models**
| Model | Test Accuracy (%) |
|--------|-----------------|
| **Soft Voting Classifier** | **80.6%** |
| **Hard Voting Classifier** | 79.0% |
| **Best Random Forest (Tuned)** | **85.45%** |
| **Best Decision Tree (Tuned)** | 84.45% |
| **Bagging SVM** | 78.75% |
| **XGBoost** | **84.95%** |
| **LightGBM** | **84.95%** |
| **Stacking Classifier** | **85.45%** |

### **Key Observations**
1. **Stacking Classifier and Random Forest achieved the highest accuracy (85.45%).**
   - Combining multiple models via stacking **did not outperform** the best base learner (**Random Forest**).
   - Stacking adds computational complexity without significant gains.

2. **Boosting Methods (XGBoost & LightGBM) performed slightly worse than Random Forest.**
   - **84.95% accuracy** suggests that boosting methods are nearly as effective but require more tuning.

3. **Soft Voting outperformed Hard Voting.**
   - **Soft Voting (80.6%) was better than Hard Voting (79%)**, indicating that using probabilities instead of majority voting improves performance.

4. **Bagging SVM was the weakest ensemble model.**
   - **78.75% accuracy** suggests that SVM is not the best base learner for bagging.

---

## **7. Results Pointers (2-3 Key Findings)**
1. **Random Forest is the best individual model (85.45% accuracy).**  
   - It outperformed all other base classifiers.
   - **Stacking with multiple models did not improve accuracy further.**

2. **XGBoost & LightGBM are strong alternatives to Random Forest.**  
   - Both **achieved 84.95% accuracy**, showing competitive performance with ensemble methods.

3. **Soft Voting works better than Hard Voting.**  
   - Probabilistic averaging (Soft Voting) gives more **robust predictions** than a majority rule (Hard Voting).

---

## **8. Tech Stack & Algorithms Used**
### **Libraries & Tools**
- **Pandas, NumPy** - Data handling
- **Scikit-learn** - Model training & evaluation
- **XGBoost, LightGBM** - Boosting models
- **Matplotlib, Seaborn** - Data visualization

### **Machine Learning Models**
- **Voting Classifier (Soft & Hard)**
- **Bagging (SVM Base Model)**
- **Boosting (XGBoost & LightGBM)**
- **Stacking (Logistic Regression Meta-Learner)**
- **Random Forest, AdaBoost, Gradient Boosting**

### **Evaluation Metrics**
- **Accuracy**
- **Hyperparameter tuning using GridSearchCV**
- **Out-of-Bag Score (Bagging Models)**

---

## **Final Conclusion**
1. **Random Forest and Stacking achieved the highest accuracy (85.45%).**  
   - Stacking did **not significantly improve over Random Forest**, suggesting that a well-tuned individual model can be just as effective.

2. **Boosting models (XGBoost & LightGBM) performed well (84.95%) but did not surpass Random Forest.**  
   - Gradient Boosting methods might need further tuning for better performance.

3. **Bagging (SVM) underperformed (78.75%).**  
   - SVM **is not the best base learner** for Bagging in this dataset.

4. **Soft Voting is more effective than Hard Voting.**  
   - Using probabilities improves ensemble predictions.

5. **Ensemble learning enhances prediction accuracy but does not always outperform the best individual model.**  
   - Random Forest alone **achieved the highest accuracy** despite the complexity of other ensemble techniques.

![image](https://github.com/user-attachments/assets/83c728c2-328f-4d99-9e4b-54580ac9b28e)
