### **Project Report: Dimension Reduction and Clustering for Categorical and Mixed Data in Housing Price Prediction**

---

## **1. Suitable Name for the Project**
**Handling Categorical Data for Dimensionality Reduction and Clustering in Housing Prices**

## **2. Project Title**
**Applying MCA, PCA, and K-Medoids with Gower Distance on Housing Price Data**

---

## **3. Project Description**
This project focuses on **dimension reduction and clustering** for **categorical and mixed data**. Traditional methods like PCA are not effective for categorical data, so **Multiple Correspondence Analysis (MCA)** is used for categorical features, and **Principal Component Analysis (PCA)** is applied to numerical features.  

For **clustering**, we use **K-Medoids with Gower Distance**, which is designed for datasets containing both numerical and categorical features.  

The dataset consists of housing prices and their corresponding features, including categorical variables like house type and neighborhood. The goal is to:
1. **Reduce the dimensionality** of categorical and numerical features.
2. **Predict house prices** using Ridge Regression with reduced features.
3. **Perform clustering using K-Medoids and evaluate its alignment with price groups.**

---

## **4. Overview Description with Key Steps**
### **Part 1: Dimension Reduction and Regression**
1. **Preprocessing**
   - Load the dataset, remove ID column and excessive missing values.
   - Split into categorical and numerical features.

2. **Ensuring Categorical Feature Consistency**
   - Match categorical levels in train and test sets to avoid MCA errors.

3. **Applying Dimensionality Reduction**
   - **Standardize numerical features** and apply **PCA (35 components)**.
   - **Apply MCA** on categorical features.

4. **Regression with Ridge Model**
   - Combine **PCA and MCA-transformed features**.
   - Fit **Ridge Regression** with **α=0.5**.
   - Evaluate model performance using **Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)**.

5. **Baseline Model with Original Features**
   - Create **dummy variables for categorical features**.
   - Fit **Ridge Regression on full data**.
   - Compare performance with the dimensionally reduced dataset.

---

### **Part 2: Clustering Analysis**
1. **Compute Gower Distance**
   - Gower metric handles **mixed numerical and categorical data**.
   - Compute **pairwise distances** between observations.

2. **Apply K-Medoids Clustering**
   - Choose **k (number of clusters) from 2 to 10**.
   - Tune **k using cross-validation with Normalized Mutual Information (NMI)**.
   - Determine the **optimal number of clusters**.

3. **Evaluating Clustering Performance**
   - Bin **SalePrice** into **quantile-based groups**.
   - Compare cluster assignments with binned SalePrice using **NMI Score**.

4. **Visualization**
   - **Elbow plot** for NMI Score vs. number of clusters.
   - **Cluster assignments visualized using PCA-reduced Gower distances**.

---

## **5. Goals**
### **Primary Goal**
- **Develop a dimensionality reduction and clustering pipeline** for mixed data types in real estate price prediction.

### **Secondary Goals**
1. **Evaluate whether dimensionality reduction improves regression performance.**
2. **Assess clustering quality using Normalized Mutual Information (NMI).**
3. **Compare the effectiveness of PCA + MCA vs. full feature set in Ridge Regression.**
4. **Optimize the number of clusters in K-Medoids using Gower distance.**

---

## **6. Results**
### **Ridge Regression Performance**
| Model | MSE | RMSE |
|--------|----------|----------|
| **Ridge Regression (Reduced Features: PCA + MCA)** | **3630.04** | **60.25** |
| **Ridge Regression (Full Features with Dummies)** | 6646.60 | 81.53 |

- **Using PCA + MCA reduced error significantly compared to full feature regression.**
- **Dimensionality reduction improves generalization, preventing overfitting.**

---

### **K-Medoids Clustering Results**
| Number of Clusters | NMI Score |
|----------------|-----------|
| 2 | **0.40** |
| 3 | 0.37 |
| 4 | 0.34 |
| 5 | 0.32 |
| 6 | 0.30 |

- **Optimal number of clusters: 2** (Highest NMI score of **0.40**).
- **Higher K values result in lower NMI scores**, indicating over-segmentation.

---

### **Visualization Results**
1. **Elbow Plot: NMI Score vs. Number of Clusters**
   - NMI **drops as K increases**, confirming **two clusters are optimal**.

2. **PCA Projection of Gower Distances**
   - Cluster assignments are well-separated in PCA space.
   - **SalePrice categories do not strongly align with clusters**, possibly due to hidden features.

---

## **7. Results Pointers (2-3 Key Findings)**
1. **Dimensionality Reduction Improves Regression Performance**
   - PCA + MCA **reduced MSE from 6646.60 to 3630.04**.
   - **Reduces noise from redundant categorical features**.

2. **K-Medoids Clustering Shows Weak Alignment with Price**
   - **NMI Score of 0.40 suggests limited correlation** between clusters and price groups.
   - **SalePrice is influenced by factors beyond categorical attributes**.

3. **Optimal Number of Clusters is 2**
   - **Increasing clusters beyond 2 reduces clustering quality**.
   - **More clusters introduce unnecessary complexity**.

---

## **8. Tech Stack & Algorithms Used**
### **Libraries & Tools**
- **Pandas, NumPy** - Data processing.
- **Scikit-learn** - Ridge Regression, PCA, StandardScaler.
- **Prince** - MCA (Multiple Correspondence Analysis).
- **Scikit-learn-extra** - K-Medoids clustering.
- **Gower** - Computing Gower Distance.

### **Machine Learning Models**
- **Ridge Regression** (for predicting SalePrice).
- **PCA (Numerical Features)**.
- **MCA (Categorical Features)**.
- **K-Medoids Clustering (Mixed Data)**.

### **Evaluation Metrics**
- **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)** for regression.
- **Normalized Mutual Information (NMI)** for clustering.

---

## **Final Conclusion**
1. **Dimensionality reduction (PCA + MCA) significantly improved regression performance**.
   - Lower **MSE (3630.04)** and **RMSE (60.25)** compared to full feature Ridge Regression.
   - **Avoids overfitting** while preserving predictive power.

2. **K-Medoids with Gower Distance effectively handles mixed data, but clusters do not align strongly with SalePrice.**
   - Best **NMI score (0.40)** found at **K=2**.
   - **Housing price prediction requires more than categorical grouping**.

3. **PCA and MCA are effective for reducing complexity without losing performance.**
   - **Lower dimensional features improved model generalization.**
   - **MCA enables categorical data transformation** for numerical models.

4. **Clustering with K-Medoids provides insights but does not fully explain price segmentation.**
   - **Cluster formation is more influenced by categorical attributes**.
   - **SalePrice segmentation requires richer feature engineering**.

---

## **Future Improvements**
1. **Incorporate additional features (e.g., location, neighborhood trends).**
2. **Experiment with different regression models (e.g., Gradient Boosting, Neural Networks).**
3. **Test alternative clustering techniques (e.g., Hierarchical Clustering, DBSCAN).**
