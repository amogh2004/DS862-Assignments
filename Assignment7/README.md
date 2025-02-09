### **Project Report: Sentiment Analysis on Yelp Reviews Using Naive Bayes**

## **1. Suitable Name for the Project**
**Yelp Sentiment Analysis Using Naive Bayes Classification**

## **2. Project Title**
**Text Classification for Yelp Reviews: Comparing Bag-of-Words and TF-IDF with Naive Bayes Models**

## **3. Project Description**
This project explores **sentiment classification** on **Yelp reviews** using **Naive Bayes classifiers**. The dataset consists of reviews labeled as **positive (1) or negative (0)**.  
The goal is to evaluate **Bag-of-Words (CountVectorizer)** and **TF-IDF (Term Frequency-Inverse Document Frequency)** feature extraction techniques and compare **Multinomial Naive Bayes** with **Gaussian Naive Bayes** for sentiment analysis.  

We optimize model performance using **GridSearchCV** to tune hyperparameters such as **alpha (smoothing parameter) for MultinomialNB** and **var_smoothing for GaussianNB**.

---

## **4. Overview Description with Key Steps**
The project follows these steps:

### **Step 1: Data Preprocessing**
- Load **Yelp labeled reviews** from the dataset.
- Remove **missing values** (if any).
- Preprocess text:
  - Convert text to **lowercase**.
  - Remove **stopwords** using **NLTK**.
  - Tokenize and clean the data.

### **Step 2: Train-Test Split**
- Split data into **80% training, 20% testing**.

### **Step 3: Feature Engineering**
- Convert text into numerical representations using:
  - **Bag-of-Words (CountVectorizer)**
  - **TF-IDF (TfidfVectorizer)**

### **Step 4: Model Training and Evaluation**
1. **Multinomial Naive Bayes (MNB)**
   - Train using **Bag-of-Words** and **TF-IDF** separately.
   - Optimize **alpha** (smoothing parameter) using **GridSearchCV**.
   - Evaluate model using **accuracy, precision, recall, and F1-score**.

2. **Gaussian Naive Bayes (GNB)**
   - Train using **TF-IDF** (converted to an array for compatibility).
   - Optimize **var_smoothing** using **GridSearchCV**.
   - Compare performance with Multinomial Naive Bayes.

---

## **5. Goals**
### **Primary Goal**
- Develop a **Naive Bayes-based sentiment classification model** for Yelp reviews.

### **Secondary Goals**
1. **Compare Feature Extraction Methods**
   - Evaluate **Bag-of-Words vs. TF-IDF** performance.

2. **Compare Different Naive Bayes Models**
   - Assess **MultinomialNB vs. GaussianNB** for text classification.

3. **Optimize Model Performance**
   - Tune **alpha (for MNB)** and **var_smoothing (for GNB)** using **GridSearchCV**.

---

## **6. Results**
### **Performance Comparison of Naive Bayes Models**
| Model | Feature Extraction | Best Hyperparameter | Test Accuracy |
|--------|-------------------|--------------------|--------------|
| **Multinomial Naive Bayes** | **Bag-of-Words** | alpha = 10 | **75.5%** |
| **Multinomial Naive Bayes** | **TF-IDF** | alpha = 20 | **79.0%** |
| **Gaussian Naive Bayes** | **TF-IDF** | var_smoothing = 0.01 | **71.0%** |

### **Key Observations**
1. **Multinomial Naive Bayes (TF-IDF) performed best (79% accuracy).**
   - Outperformed **Bag-of-Words (75.5%)** by capturing important term importance.
   - Performed **better than GaussianNB**, which assumes continuous features.

2. **Multinomial Naive Bayes is the best choice for text classification.**
   - **TF-IDF improved performance** over Bag-of-Words by reducing noise.
   - **Higher recall for negative reviews (84%)** ensures better sentiment detection.

3. **Gaussian Naive Bayes struggled with text data.**
   - **71% accuracy** indicates it is not the best model for discrete text-based classification.
   - Assumes continuous probability distribution, which does not fit sparse text features.

---

## **7. Results Pointers (2-3 Key Findings)**
1. **TF-IDF Outperforms Bag-of-Words for Sentiment Analysis.**  
   - TF-IDF captures **important words while reducing noise** from frequent terms.
   - **4% improvement in accuracy** over Bag-of-Words.

2. **Multinomial Naive Bayes is Ideal for Text Classification.**  
   - Achieved the **best balance between precision and recall**.
   - **TF-IDF with MNB (79%) outperforms GaussianNB (71%)**.

3. **Gaussian Naive Bayes is Not Suitable for Text Data.**  
   - Lower accuracy (**71%**) suggests **GNB is better for continuous numerical data**.
   - Text data is sparse and discrete, making **MultinomialNB a better choice**.

---

## **8. Tech Stack & Algorithms Used**
### **Libraries & Tools**
- **Pandas, NumPy** - Data manipulation
- **NLTK (Natural Language Toolkit)** - Stopword removal
- **Scikit-learn** - Model training, vectorization, and evaluation

### **Machine Learning Models**
- **Multinomial Naive Bayes (MNB)**
- **Gaussian Naive Bayes (GNB)**

### **Feature Engineering Techniques**
- **Bag-of-Words (CountVectorizer)**
- **TF-IDF (TfidfVectorizer)**

### **Evaluation Metrics**
- **Accuracy**
- **Precision, Recall, and F1-score**
- **Hyperparameter tuning using GridSearchCV**

---

## **Final Conclusion**
1. **Multinomial Naive Bayes with TF-IDF is the best model (79% accuracy).**  
   - Captures sentiment **better than Bag-of-Words (75.5%)**.
   - **Higher recall (84%)** ensures **better detection of negative reviews**.

2. **Gaussian Naive Bayes does not perform well on text data.**  
   - **71% accuracy** shows it struggles with sparse and discrete text features.

3. **Naive Bayes is a strong baseline model for text classification.**  
   - Despite its simplicity, **MNB achieves competitive performance**.

4. **Further Improvements Could Include:**
   - **N-grams (bigram, trigram features)** to capture phrase-level sentiment.
   - **Word embeddings (Word2Vec, BERT, FastText)** for deep learning-based text classification.
![Uploading image.png…]()
