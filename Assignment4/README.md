### **Project Report: Comparing MLP and Gradient Boosting for Regression and Classification**

## **1. Suitable Name for the Project**
**MLP vs. Gradient Boosting: A Comparative Analysis for Regression and Classification**

## **2. Project Title**
**Neural Networks vs. Gradient Boosting: Evaluating MLP Architectures on Regression and Classification Tasks**

## **3. Project Description**
This project evaluates different **Multi-Layer Perceptron (MLP) architectures** and compares them with **Gradient Boosting** models for **regression and classification tasks**.  
- **For regression**, we use the **California Housing dataset** to predict median house values based on various factors.  
- **For classification**, we use the **Mobile Price Classification dataset**, predicting whether a mobile device falls into a "low price" or "high price" category based on its technical specifications.  

Through a series of experiments, we optimize model structures, test different MLP depths, and compare their performance against **Gradient Boosting** to determine the most effective approach.

---

## **4. Overview Description with Key Steps**
The project workflow consists of the following steps:

### **Regression Task (California Housing Dataset)**
1. **Data Preprocessing**
   - Load the dataset and normalize features using **StandardScaler**.
   - Split into **training (60%), validation (20%), and testing (20%)**.

2. **Building & Evaluating MLP Models**
   - Train two different **MLP architectures**:
     - **Shallow Network** (2 hidden layers: 15 & 10 neurons).
     - **Deep Network** (5 hidden layers: 7, 5, 3, 2, 2 neurons).
   - Evaluate performance using **Mean Squared Error (MSE)**.

3. **Comparing with Gradient Boosting Regressor**
   - Train a **tuned Gradient Boosting Regressor**.
   - Compare performance using **MSE & R² score**.

---

### **Classification Task (Mobile Price Dataset)**
4. **Data Preprocessing**
   - Convert categorical labels into binary (low vs. high price).
   - Standardize numerical features.
   - Split data into **training (60%), validation (20%), and testing (20%)**.

5. **Building & Evaluating MLP Models**
   - Train an **MLP classifier** with:
     - 3 hidden layers: **64, 32, 16 neurons** (ReLU activation).
   - Evaluate performance using **accuracy and MSE**.

6. **Comparing with Gradient Boosting Classifier**
   - Train a **tuned Gradient Boosting Classifier**.
   - Compare performance using **accuracy and MSE**.

---

## **5. Goals**
### **Primary Goal**
- Evaluate **MLP architectures** for both **regression and classification** and compare their performance with **Gradient Boosting models**.

### **Secondary Goals**
1. **Identify the best MLP structure for regression and classification.**
   - Compare different layer depths and neuron distributions.

2. **Compare Neural Networks with Gradient Boosting.**
   - Determine whether **ensemble methods** outperform deep learning for structured tabular data.

3. **Optimize Performance Metrics.**
   - **Regression:** Use **Mean Squared Error (MSE)** and **R² score**.
   - **Classification:** Use **Accuracy** and **Mean Squared Error (MSE)**.

---

## **6. Results**
### **Regression Task**
- **Shallow MLP (15 & 10 neurons) performed better (MSE = 0.286).**  
  - Generalized well to test data.  
- **Deeper MLP (5 layers) had higher MSE (1.373), indicating overfitting.**
- **Gradient Boosting Regressor outperformed both MLPs with MSE = 0.2451.**  
  - Achieved **R² = 0.8213**, indicating strong predictive performance.

---

### **Classification Task**
- **MLP achieved 96.50% accuracy** with an **MSE of 0.035**.
- **Gradient Boosting Classifier slightly outperformed MLP, achieving 97.25% accuracy** and **MSE of 0.0275**.
- Both models performed well, but **Gradient Boosting was slightly more accurate**.

---

## **7. Results Pointers (2-3 Key Findings)**
1. **Gradient Boosting Regressor outperformed MLP for regression.**  
   - Lower **MSE (0.2451)** and higher **R² (0.8213)** suggest it captured patterns better.  
   - MLP struggled with deeper networks, likely due to overfitting.

2. **Both MLP and Gradient Boosting performed well in classification, but GBC was slightly better.**  
   - **Gradient Boosting achieved 97.25% accuracy**, compared to **96.50% for MLP**.
   - **GBC had lower MSE (0.0275)**, indicating better class separation.

3. **Shallow MLP models generalized better than deeper ones.**  
   - **Regression:** Fewer neurons led to better generalization.
   - **Classification:** A well-tuned MLP performed almost as well as GBC.

---

## **8. Tech Stack & Algorithms Used**
### **Libraries & Tools**
- **Python (TensorFlow, Keras, Scikit-learn, Pandas, NumPy, Matplotlib)**

### **Machine Learning Models**
- **Multi-Layer Perceptron (MLP)**
  - Regression (California Housing)
  - Classification (Mobile Price)
- **Gradient Boosting Models**
  - **Gradient Boosting Regressor (GBR)**
  - **Gradient Boosting Classifier (GBC)**

### **Evaluation Metrics**
- **Regression:**
  - **Mean Squared Error (MSE)**
  - **R² Score**
- **Classification:**
  - **Accuracy**
  - **Confusion Matrix**
  - **Mean Squared Error (MSE)**

---

### **Final Conclusion**
1. **For Regression, Gradient Boosting performed significantly better than MLP.**  
   - It achieved the lowest MSE and highest R².  
   - Deep MLP architectures struggled with overfitting.  

2. **For Classification, both MLP and Gradient Boosting performed well, but GBC was slightly better.**  
   - Achieved **higher accuracy** and **lower MSE**.

3. **MLP can be effective, but structured tabular data often favors Gradient Boosting methods.**  
   - Tuning deep networks is challenging, whereas Gradient Boosting adapts more efficiently.
![image](https://github.com/user-attachments/assets/253785fa-74da-4033-84cc-60e3156e93f2)
