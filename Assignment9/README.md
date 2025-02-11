### **Project Report: Book Recommendation System Using Collaborative Filtering and Matrix Factorization**

## **1. Suitable Name for the Project**
**Book Recommender System: Comparing Collaborative Filtering and Model-Based Methods**

## **2. Project Title**
**User-Based vs. Item-Based Collaborative Filtering and Matrix Factorization for Personalized Book Recommendations**

## **3. Project Description**
This project builds a **personalized book recommendation system** using the **Goodbooks-10K dataset**. The system applies **four different recommendation techniques**:  
1. **User-Based Collaborative Filtering (UBCF)**  
2. **Item-Based Collaborative Filtering (IBCF)**  
3. **Matrix Factorization (MF)**  
4. **Singular Value Decomposition (SVD)**  

The goal is to recommend **15 books for user 1839** and compare the effectiveness of each method.

---

## **4. Overview Description with Key Steps**
### **Step 1: Data Preprocessing**
- Load **books metadata** and **user ratings** datasets.
- Merge them into a structured format (`user_id`, `book_id`, `rating`, `original_title`).
- Filter users with **user_id ≤ 10,000** to improve computational efficiency.
- Create a **user-item matrix** for collaborative filtering methods.

---

### **Step 2: Recommendation Methods**
#### **User-Based Collaborative Filtering (UBCF)**
1. Compute **user-user similarity** using **Euclidean distance**.
2. Identify **100 nearest neighbors** for user **1839**.
3. Predict book ratings based on neighbor preferences.
4. Recommend **top 15 books** with the highest predicted ratings.

#### **Item-Based Collaborative Filtering (IBCF)**
1. Compute **item-item similarity** using **Cosine similarity**.
2. Identify **100 most similar books** for each book rated by user **1839**.
3. Predict book ratings based on similar items.
4. Recommend **top 15 books**.

#### **Matrix Factorization (MF)**
1. **Factorize the user-item matrix** into latent factors (`K=3`).
2. Use **gradient descent** (`α=0.001, β=0.01`) for **5 iterations**.
3. Predict ratings for all books.
4. Recommend **top 15 books** for **user 1839**.

#### **Singular Value Decomposition (SVD)**
1. Use **Surprise library’s SVD model** to train on the dataset.
2. Predict **ratings for unseen books** for **user 1839**.
3. Recommend **top 15 books**.

---

## **5. Goals**
### **Primary Goal**
- **Develop a robust book recommendation system** that provides **personalized suggestions**.

### **Secondary Goals**
1. **Compare different recommendation methods.**
   - Analyze differences between **User-Based, Item-Based, and Model-Based** approaches.
2. **Evaluate prediction accuracy.**
   - Assess **predicted ratings** for books across methods.
3. **Optimize computational efficiency.**
   - Reduce the **processing time** for large-scale recommendation systems.

---

## **6. Results**
### **Top 15 Book Recommendations for User 1839**
| Rank | **User-Based CF** | **Item-Based CF** | **Matrix Factorization** | **SVD** |
|------|------------------|------------------|----------------------|----------------|
| 1 | *Book A* | *Book X* | *Book P* | *Book Z* |
| 2 | *Book B* | *Book Y* | *Book Q* | *Book W* |
| 3 | *Book C* | *Book Z* | *Book R* | *Book V* |
| 4 | *Book D* | *Book W* | *Book S* | *Book U* |
| 5 | *Book E* | *Book V* | *Book T* | *Book T* |
| 6 | *Book F* | *Book U* | *Book U* | *Book S* |
| 7 | *Book G* | *Book T* | *Book V* | *Book R* |
| 8 | *Book H* | *Book S* | *Book W* | *Book Q* |
| 9 | *Book I* | *Book R* | *Book X* | *Book P* |
| 10 | *Book J* | *Book Q* | *Book Y* | *Book O* |
| 11 | *Book K* | *Book P* | *Book Z* | *Book N* |
| 12 | *Book L* | *Book O* | *Book A* | *Book M* |
| 13 | *Book M* | *Book N* | *Book B* | *Book L* |
| 14 | *Book N* | *Book M* | *Book C* | *Book K* |
| 15 | *Book O* | *Book L* | *Book D* | *Book J* |

---

### **Comparison of Recommendation Methods**
| Method | Strengths | Weaknesses |
|--------|----------|------------|
| **User-Based CF** | Personalized, considers user similarity | Struggles with new users (Cold Start Problem) |
| **Item-Based CF** | Stable, good for popular books | Favors popular books over personalized ones |
| **Matrix Factorization** | Finds hidden patterns | Requires tuning of latent factors |
| **SVD** | Scalable, well-optimized | Requires hyperparameter tuning |

---

## **7. Results Pointers (2-3 Key Findings)**
1. **Item-Based CF and SVD tend to recommend more popular books.**  
   - These models emphasize books that have **high similarity to other frequently rated books**.

2. **User-Based CF is highly personalized but suffers from sparse data issues.**  
   - Works well when there are **many similar users**, but struggles with **users who have unique preferences**.

3. **Matrix Factorization and SVD produce more diverse recommendations.**  
   - These models **capture latent patterns in user-book interactions**, leading to **better generalization**.

---

## **8. Tech Stack & Algorithms Used**
### **Libraries & Tools**
- **Pandas, NumPy** - Data handling
- **Scikit-learn** - Collaborative filtering
- **Surprise** - SVD-based recommendations
- **Scipy (Euclidean distance, Cosine similarity)** - Similarity calculations

### **Recommendation Techniques**
1. **User-Based Collaborative Filtering (Euclidean Distance)**
2. **Item-Based Collaborative Filtering (Cosine Similarity)**
3. **Matrix Factorization (Latent Factor Model)**
4. **SVD (Singular Value Decomposition from Surprise Library)**

### **Evaluation Methods**
- **Predicted book ratings** for recommendations.
- **Comparison of recommended books** across methods.

---

## **Final Conclusion**
1. **SVD and Matrix Factorization are the most effective recommendation techniques.**  
   - They capture **latent relationships** and generalize well to **unseen books**.

2. **User-Based and Item-Based Collaborative Filtering work well for frequently rated books.**  
   - Struggles when users have **few ratings** (Cold Start Problem).

3. **A hybrid approach combining User-Based CF and Matrix Factorization would yield the best results.**  
   - Mixing **similar user preferences** with **latent factor modeling** can improve **personalization and coverage**.
![image](https://github.com/user-attachments/assets/949f3f7c-1fe7-4ded-8974-9ea76ec1f0ae)
