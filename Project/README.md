### **Final Project Report: The Influence of Credit Limit Variability on Credit Score Tiers**

---

## **1. Suitable Name for the Project**
**Impact of Credit Limit Changes on Credit Score Categories: A Machine Learning Study**

## **2. Project Title**
**The Influence of Credit Limit Variability on Credit Score Tiers: A Statistical Study**

---

## **3. Project Description**
This project aims to analyze how **percentage changes in credit limits** influence **credit score categories** (**Poor, Standard, Good**) and the role of financial and behavioral factors in determining creditworthiness.  

The dataset consists of **100,000 records and 28 features**, with a focus on **credit utilization, payment history, debt levels, and credit limit changes**.  

We apply **classification models** to predict **credit scores** and use **unsupervised clustering (K-Means)** to uncover hidden behavioral patterns.

---

## **4. Overview Description with Key Steps**
### **Step 1: Data Preprocessing**
- **Dropped irrelevant columns** (`id`, `customer_id`, `month`, `name`, `ssn`, `occupation`, `type_of_loan`, `payment_behaviour`).
- **Categorical feature encoding**: `credit_mix`, `payment_of_min_amount` mapped to numerical values.
- **Binned numerical features** (`num_bank_accounts`, `num_credit_card`, `num_of_loan`).
- **Feature engineering**:
  - Converted **credit_history_age** to **months**.
  - Cleaned **num_of_loan** and **outstanding_debt** for uniformity.

---

### **Step 2: Hypothesis Testing**
#### **Hypothesis**
- **Null Hypothesis (H0):** Changes in credit limits **do not** significantly impact credit scores.
- **Alternative Hypothesis (H1):** Changes in credit limits **do** significantly affect credit scores.

#### **Findings**
- ANOVA tests revealed that `changed_credit_limit` has a statistically significant effect (**p < 0.001**).
- However, feature importance analysis suggested that **other factors play a larger role**.

---

### **Step 3: Feature Selection and Dimensionality Reduction**
- **Chi-Square Test**: Confirmed significance of categorical features in determining credit scores.
- **ANOVA Test**: Identified key numerical predictors.
- **Cramér’s V**: Analyzed categorical variable relationships.
- **PCA Consideration**: Not used due to the dataset’s interpretability needs.

---

### **Step 4: Machine Learning Models for Credit Score Classification**
| Model | Hyperparameter Optimization | Accuracy (%) |
|--------|----------------------------|--------------|
| **Decision Tree** | `max_depth=30, min_samples_split=2` | 77% |
| **K-Nearest Neighbors (KNN)** | `n_neighbors=9, metric=manhattan` | 81% |
| **Random Forest** | `n_estimators=200, max_depth=None` | **83%** |
| **Gradient Boosting (CatBoost)** | `iterations=500, depth=8` | 78% |
| **Neural Network (MLP)** | `hidden_layers=(100, 50), activation=tanh` | **85%** |

- **Random Forest and Neural Networks performed best**.
- **Decision Trees and CatBoost showed lower accuracy** but provided good interpretability.
- **KNN was a strong baseline model but computationally expensive**.

---

### **Step 5: Unsupervised Clustering with K-Means**
1. **Applied K-Means (k=3, based on silhouette score) to segment customers**.
2. **Mapped clusters to credit score categories**.
3. **Found strong alignment** between clusters and high-risk vs. low-risk credit profiles.

---

## **5. Goals**
### **Primary Goal**
- **Predict credit score categories accurately** based on financial and credit behavior.

### **Secondary Goals**
1. **Evaluate the influence of `changed_credit_limit` on credit scores**.
2. **Compare performance of different classification models**.
3. **Use clustering to uncover hidden patterns in credit behavior**.

---

## **6. Results**
### **Key Findings**
1. **Credit Limit Changes Have an Effect, but Other Features Are More Important**
   - ANOVA suggests statistical significance, but **features like outstanding debt, credit mix, and credit history age have a greater influence**.

2. **Random Forest and Neural Networks Are the Best Predictive Models**
   - **Neural Network (MLP) performed the best (85% accuracy)**.
   - **Random Forest was the best interpretable model (83% accuracy)**.

3. **Clustering Confirms Hidden Patterns in Credit Behavior**
   - **Customers with high credit utilization and frequent delays were grouped into high-risk clusters**.

---

## **7. Results Pointers (2-3 Key Findings)**
1. **The most important factors influencing credit scores are**:
   - **Outstanding Debt**
   - **Credit Mix**
   - **Credit History Age**
   - **Interest Rate**
   - **Delay from Due Date**

2. **Machine learning models can effectively classify credit scores with up to 85% accuracy**.
   - **Ensemble methods (Random Forest, CatBoost) and Neural Networks** perform the best.
   - **KNN and Decision Trees offer interpretable insights but lower accuracy**.

3. **Unsupervised clustering provides insights into customer segmentation**.
   - **High-risk customers have frequent delayed payments and high outstanding debt**.

---

## **8. Tech Stack & Algorithms Used**
### **Libraries & Tools**
- **Pandas, NumPy** - Data handling
- **Scikit-learn** - Machine learning models
- **Seaborn, Matplotlib** - Visualization
- **Scipy, Statsmodels** - Statistical tests
- **Captum** - Neural network feature importance

### **Machine Learning Models**
- **Decision Tree**
- **K-Nearest Neighbors (KNN)**
- **Random Forest**
- **Gradient Boosting (CatBoost)**
- **Neural Network (MLP)**

### **Evaluation Metrics**
- **Accuracy, Precision, Recall, F1-Score** (for classification)
- **Normalized Mutual Information (NMI)** (for clustering)

---

## **Final Conclusion**
1. **Credit Limit Variability Matters, But It's Not the Primary Factor in Credit Scores**
   - **More significant predictors** include **credit history age, outstanding debt, and interest rate**.
   - Financial institutions should **prioritize long-term financial behavior** over short-term credit limit changes.

2. **Machine Learning Models Can Accurately Predict Credit Scores**
   - **Neural Networks achieved 85% accuracy**, making them suitable for **automated credit risk assessment**.
   - **Random Forests (83%) provide strong interpretability** for financial institutions.

3. **Clustering Provides Business Insights for Credit Risk Management**
   - Customers can be **segmented based on risk profiles**.
   - Banks can use **personalized credit limit adjustments** based on risk clusters.

---

## **Future Improvements**
1. **Incorporate more behavioral data** (e.g., spending patterns, transaction history).
2. **Test deep learning models (e.g., Transformer-based networks)** for better accuracy.
3. **Use Explainable AI (XAI) techniques** to improve model interpretability.

![Uploading image.png…]()
