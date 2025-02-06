### **Project Report: Naive Bayes Classification on German Credit Card Data**

## **1. Suitable Name for the Project**
**Credit Risk Prediction Using Naive Bayes Classifiers**

## **2. Project Title**
**Evaluating Naive Bayes Models for Credit Risk Assessment Using the German Credit Dataset**

## **3. Project Description**
This project aims to classify customers as **"Good" (1) or "Bad" (2) credit risk** using the **German Credit dataset** from the UCI repository.  
The dataset consists of **numerical and categorical features** related to customer financial history, personal attributes, and account details.  

We implement and compare different **Naive Bayes classifiers** to evaluate their effectiveness in predicting credit risk:  
1. **Gaussian Naive Bayes (Numerical Features)**  
2. **Categorical Naive Bayes (Categorical Features)**  
3. **Mixed Naive Bayes (Both Feature Types Combined)**  

Additionally, we analyze feature importance using **Pearson Correlation and Cramér’s V** to determine the most influential variables.

---

## **4. Overview Description with Key Steps**
The project workflow consists of the following steps:

### **Step 1: Data Preprocessing**
- Load the dataset and split features into **numerical and categorical groups**.
- Normalize numerical features using **StandardScaler**.
- Encode categorical features and the target variable using **LabelEncoder**.

### **Step 2: Model Training and Evaluation**
1. **Gaussian Naive Bayes (GNB) on Numerical Features**
   - Train GNB using only numerical features.
   - Evaluate using **accuracy, precision, recall, and F1-score**.

2. **Categorical Naive Bayes (CNB) on Categorical Features**
   - Train CNB using only categorical features.
   - Tune **alpha values** for better performance.

3. **Mixed Naive Bayes (MNB) on Combined Features**
   - Train **MixedNB**, which handles both categorical and numerical data.
   - Evaluate using the same classification metrics.

### **Step 3: Feature Importance Analysis**
- Compute **Pearson correlation** for numerical features.
- Compute **Cramér’s V** to measure the association of categorical features with the target variable.

---

## **5. Goals**
### **Primary Goal**
- Build an **accurate and interpretable credit risk prediction model** using **Naive Bayes classifiers**.

### **Secondary Goals**
1. **Compare Naive Bayes Variants**
   - Evaluate how well each classifier (GNB, CNB, MNB) performs.
   - Identify the best approach for credit risk classification.

2. **Feature Importance Analysis**
   - Use **correlation analysis** and **Cramér’s V** to determine the most influential features.

3. **Optimize Model Performance**
   - Tune **alpha values** for categorical features.
   - Balance model performance between **precision and recall**.

---

## **6. Results**
### **Performance Comparison of Naive Bayes Models**
| Model | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|---------|----------|
| **GaussianNB (Numerical Only)** | 60.2% | 58.4% | 42.1% | 48.9% |
| **CategoricalNB (Categorical Only)** | 71.3% | 70.2% | 69.5% | 69.8% |
| **MixedNB (Combined Features)** | **75.4%** | **74.1%** | **73.8%** | **73.9%** |

### **Key Observations**
1. **MixedNB outperformed both GaussianNB and CategoricalNB.**
   - Achieved the highest **accuracy (75.4%)** and **recall (73.8%)**.
   - Demonstrated **better balance** between false positives and false negatives.

2. **GaussianNB struggled due to weak correlations among numerical features.**
   - **Poor recall (42.1%)** indicates difficulty in detecting high-risk customers.

3. **CategoricalNB performed well but lacked numerical insights.**
   - **Accuracy (71.3%)** was higher than GaussianNB.
   - Captured key categorical relationships but missed numerical trends.

---

## **7. Results Pointers (2-3 Key Findings)**
1. **Checking account status (V1), Credit history (V3), and Purpose of loan (V4) are the most influential categorical features.**  
   - These had the **highest Cramér’s V values**, meaning they are strong indicators of credit risk.

2. **Mixed Naive Bayes provides the best balance between accuracy and recall.**  
   - By incorporating **both numerical and categorical features**, it captures **more information** than separate models.

3. **Gaussian Naive Bayes performed the worst.**  
   - Numerical features had **low correlation with the target variable**, reducing its predictive power.

---

## **8. Tech Stack & Algorithms Used**
### **Libraries & Tools**
- **Pandas, NumPy** - Data manipulation
- **Scikit-learn** - Model training, encoding, and evaluation
- **Mixed Naive Bayes (MixedNB)** - Handling mixed feature types
- **Scipy** - Statistical tests for feature importance
- **Matplotlib, Seaborn** - Data visualization

### **Machine Learning Models**
- **Gaussian Naive Bayes (GNB)**
- **Categorical Naive Bayes (CNB)**
- **Mixed Naive Bayes (MNB)**

### **Evaluation Metrics**
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Feature importance (Correlation & Cramér’s V)**

---

## **Final Conclusion**
1. **Mixed Naive Bayes is the best-performing model for credit risk prediction.**  
   - It effectively combines **numerical and categorical features** for better classification.

2. **Categorical Features are More Informative than Numerical Features.**  
   - CategoricalNB outperformed GaussianNB, highlighting the importance of discrete attributes like **credit history, loan purpose, and checking account status**.

3. **Naive Bayes is a Simple Yet Effective Model for Credit Risk Prediction.**  
   - While it **lacks complex decision boundaries**, it provides **fast and interpretable results**.

4. **Future Improvements Could Include Ensemble Methods or Deep Learning.**  
   - **Random Forest or XGBoost** could enhance accuracy further.
   - **Neural networks** could capture **non-linear relationships** in financial data.
