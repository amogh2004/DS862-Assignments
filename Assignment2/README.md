### **Project Report: Fashion MNIST & Lena Image Clustering for Exploration and Compression**


This project investigates the use of clustering techniques in two different applications:  
1. **Clustering for Data Exploration** - Using Fashion MNIST, we group similar clothing items to understand patterns in the dataset.  
2. **Image Compression via Color Quantization** - Using the Lena image, we apply clustering to reduce the number of colors, achieving compression.

We employ **Agglomerative Clustering, Gaussian Mixture Models (GMM), and DBSCAN** for image clustering, and **K-Means** for image compression.

---

## **Overview Description with Key Steps**
The workflow of this project includes:

1. **Data Preprocessing:**
   - Load the Fashion MNIST dataset and preprocess it by removing duplicates.
   - Load the Lena image and reshape it for color quantization.

2. **Clustering for Fashion MNIST:**
   - Apply **Agglomerative Clustering, GMM, and DBSCAN** on a subset of 1500 images.
   - Visualize cluster centers as 28x28 images.
   - Compute **Rand Index** to evaluate clustering performance.

3. **DBSCAN Parameter Tuning:**
   - Use **K-distance plot** to find an optimal epsilon.
   - Test different configurations and compute **Rand Index**.

4. **Image Compression using K-Means:**
   - Apply **K-Means** on the Lena image with different values of **K**.
   - Use the **Elbow Method** to determine the optimal number of clusters.
   - Visualize compressed images for **K=4 and K=10**.

---

## **Goals**
### **Primary Goal**
- Explore clustering techniques for grouping images and color quantization.

### **Secondary Goals**
1. **Evaluate the Effectiveness of Clustering on Image Data:**  
   - Understand how different clustering methods perform on Fashion MNIST.
   - Determine if they align with human-perceived categories.

2. **Optimize Image Compression via Clustering:**  
   - Reduce the number of colors in the Lena image while maintaining visual quality.

3. **Compare Clustering Methods Based on Performance Metrics:**  
   - Measure cluster purity using the **Rand Index**.
   - Determine the best clustering approach for image grouping.

---

## **Results**
### **Key Findings**
- **Agglomerative Clustering with 10 clusters** provided the best separation in Fashion MNIST.
- **Gaussian Mixture Models (GMM)** captured meaningful cluster centers, with improved interpretability.
- **DBSCAN struggled** with Fashion MNIST, producing **moderate Rand Index scores (0.51 - 0.53)**.
- **K-Means compression (K=4 & K=10)** demonstrated a trade-off between visual quality and compression.
- **The Elbow Method identified K=4 as optimal** for balancing compression and image fidelity.

---

## **Tech Stack & Algorithms Used**
### **Libraries & Tools**
- **Pandas, Matplotlib, NumPy** - Data processing and visualization.
- **Scikit-learn** - Clustering algorithms and evaluation metrics.

### **Clustering Algorithms Used**
- **Agglomerative Clustering** (Hierarchical clustering)
- **Gaussian Mixture Models (GMM)**
- **DBSCAN** (Density-Based Spatial Clustering)
- **K-Means** (For color quantization)

### **Evaluation Metrics**
- **Rand Index** - Evaluates clustering quality.
- **Elbow Method** - Determines the optimal number of clusters for K-Means.
