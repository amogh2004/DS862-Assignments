# DS862-Assignments
Machine Learning for Business Analyst Assignments and Projects

1. Fashion MNIST Classification with Dimensionality Reduction </br>
   - Develop an effective classification model for Fashion MNIST that maximizes accuracy.
   - Evaluate Dimensionality Reduction Impact: Assess whether dimensionality reduction improves classification performance or leads to information loss.
   - Hyperparameter Optimization: Identify the best hyperparameter settings for Random Forest and KNN.
   - Comparison of Model Performance: Determine which classifier (Random Forest vs. KNN) is more effective for this dataset.
     
2. Fashion MNIST & Lena: Clustering for Pattern Discovery and Compression
   - Explore clustering techniques for grouping images and color quantization.
   - Evaluate the Effectiveness of Clustering on Image Data:
      - Understand how different clustering methods perform on Fashion MNIST.
      - Determine if they align with human-perceived categories.
   - Optimize Image Compression via Clustering: Reduce the number of colors in the Lena image while maintaining visual quality.
   - Compare Clustering Methods Based on Performance Metrics:
      - Measure cluster purity using the Rand Index.
      - Determine the best clustering approach for image grouping.
        
3. Customer Churn Prediction Using SVM & Logistic Regression
   - Build an effective churn prediction model that accurately identifies customers likely to leave.
   - Evaluate SVM Performance: Compare different SVM kernels (Linear, Polynomial, RBF) to determine the best-performing model.
   - Compare Against Logistic Regression: Investigate whether regularized logistic regression can outperform SVM models in churn prediction.
   - Optimize Model Hyperparameters: Use GridSearchCV to tune C, gamma, and kernel parameters for the best model performance.
     
4. MLP vs. Gradient Boosting: A Comparative Analysis for Regression and Classification
   - Evaluate MLP architectures for both regression and classification and compare their performance with Gradient Boosting models.
   - Identify the best MLP structure for regression and classification.
   - Compare different layer depths and neuron distributions.
   - Compare Neural Networks with Gradient Boosting.
   - Determine whether ensemble methods outperform deep learning for structured tabular data.
   - Optimize Performance Metrics.
      - Regression: Use Mean Squared Error (MSE) and R² score.
      - Classification: Use Accuracy and Mean Squared Error (MSE).

5. Naive Bayes Classification on German Credit Card Data
   - Build an accurate and interpretable credit risk prediction model using Naive Bayes classifiers.
   - Compare Naive Bayes Variants:
        - Evaluate how well each classifier (GNB, CNB, MNB) performs.
        - Identify the best approach for credit risk classification.
   - Feature Importance Analysis
      - Use correlation analysis and Cramér’s V to determine the most influential features.
      - Optimize Model Performance
   - Tune alpha values for categorical features.
   - Balance model performance between precision and recall.
     
6. Ensemble Learning for Bank Churn Prediction
   - Use ensemble learning to improve customer churn prediction accuracy.
   - Evaluate the effectiveness of different ensemble methods: Compare Voting, Bagging, Boosting, and Stacking.
   - Tune Hyperparameters for Best Performance: Optimize individual models before combining them.
   - Compare Accuracy Between Base Learners and Ensemble Models: Identify whether ensemble models outperform individual classifiers.

7. Sentiment Analysis on Yelp Reviews Using Naive Bayes
   - Develop a Naive Bayes-based sentiment classification model for Yelp reviews.
   - Compare Feature Extraction Methods: Evaluate Bag-of-Words vs. TF-IDF performance.
   - Compare Different Naive Bayes Models: Assess MultinomialNB vs. GaussianNB for text classification.
   - Optimize Model Performance: Tune alpha (for MNB) and var_smoothing (for GNB) using GridSearchCV.   


8. Uncovering News Topics: LDA vs. NMF for ABC News Headlines
   - Extract meaningful topics from ABC News headlines using LDA and NMF.
   - Compare LDA and NMF in topic extraction accuracy.
   - Analyze how different feature extraction methods (BoW vs. TF-IDF) affect results.
   - Determine the best model for news topic discovery.

11. Book Recommendation System Using Collaborative Filtering and Matrix Factorization
12. Extra Credit: License Plate Character and Digit Recognition Using CNNs
13. Midterm: Handling Categorical Data for Dimensionality Reduction and Clustering in Housing Prices
14. Project: The Influence of Credit Limit Variability on Credit Score Tiers
15. SVM
