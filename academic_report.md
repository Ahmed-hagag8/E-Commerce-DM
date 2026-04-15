# 📄 Research Report: Intelligent E-Commerce Analytics utilizing Machine Learning and Hybrid Recommendation Systems

## III. METHODOLOGY AND FEATURE ENGINEERING

To accurately predict customer churn and provide personalized recommendations, a robust feature engineering process was conducted via a Medallion Data Architecture (Bronze, Silver, Gold). For Customer Churn prediction, the RFM (Recency, Frequency, Monetary) model was utilized. Textual review data were also transformed using Natural Language Processing (VADER and TextBlob) to extract a `SentimentScore`.

### A. Class Distribution and Data Sparsity
The customer dataset exhibits a distinct class distribution based on the `ChurnLabel` target variable (defined as Recency > 90 days). Similar to typical e-commerce environments, the user-item interaction matrix for the recommendation engine presented a high sparsity level of 92.0%, which justified the necessity of a Hybrid Collaborative-Content model over a traditional Collaborative Filtering approach.

### B. Deep Learning Image Classification (CNN)
A PyTorch CNN was trained to classify clothing images into 15 categories, and its per-class performance is shown in Fig. 1.

**Fig. 1: CNN Confusion Matrix for Clothing Image Classification**

![CNN Confusion Matrix](data/generated/cnn_confusion_matrix.png)

### C. Association Rules Mining (Apriori)
The Apriori algorithm was applied on `Fact_Orders` to discover co-purchase patterns using Support, Confidence, and Lift metrics (Fig. 2).

**Fig. 2: Association Rules - Support vs Confidence vs Lift Analysis**

![Association Rules Chart](data/generated/association_rules_chart.png)

### D. Sentiment Analysis (NLP)
An ensemble of VADER and TextBlob analyzed 22,641 reviews, assigning each a Sentiment Score from -1 to +1 (Fig. 3).

**Fig. 3: Sentiment Analysis - Distribution by Rating, Category, and Polarity**

![Sentiment Analysis Charts](data/generated/sentiment_analysis_charts.png)

## IV. EXPERIMENTAL RESULTS AND EVALUATION

This section presents the statistical analysis of the generated features and evaluates the performance of the proposed machine learning models.

### A. Statistical Analysis of Customer Features
A thorough statistical analysis was conducted to understand the relationship between engineered features and the churn status. Table I illustrates the statistical differences in RFM behavior between Active and Churned customers.

**TABLE I: Statistical Analysis of RFM Features with Respect to ChurnLabel**

| Feature | Class (Churn) | Mean | Min | 50% (Median) | Max |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Recency (Days)** | 0 (Active) | 41.5 | 1 | 35.0 | 90 |
| | 1 (Churned) | 215.3 | 91 | 180.0 | 373 |
| **Frequency (Orders)**| 0 (Active) | 6.8 | 1 | 4.0 | 210 |
| | 1 (Churned) | 1.2 | 1 | 1.0 | 34 |
| **Monetary ($)** | 0 (Active) | $4,320 | $15 | $1,200 | $280K |
| | 1 (Churned) | $350 | $2 | $150 | $12K |

### B. Performance Metrics for Supervised Machine Learning
To predict customer churn, five different supervised machine learning algorithms were trained and evaluated: Logistic Regression, Support Vector Machines (SVM), Multi-Layer Perceptron (MLP), XGBoost, and Random Forest. The Recency feature was intentionally excluded from training to prevent data leakage, since the ChurnLabel is directly derived from it (Recency > 90). This forces the models to learn churn patterns from genuine behavioral features such as Frequency, Monetary value, and CustomerLifespan.

The comparative analysis (shown in Table II) reveals differentiated performance across all five algorithms.

**TABLE II: Machine Learning Models Results for Churn Prediction**

| Metric | Logistic Regression | Random Forest | **XGBoost** | **SVM** | MLP Neural Net |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 0.7211 | 0.7270 | **0.7389** | 0.7313 | 0.7253 |
| **Precision** | 0.7045 | 0.7134 | 0.7118 | 0.6895 | 0.7013 |
| **Recall** | 0.7776 | 0.7742 | 0.8177 | **0.8579** | 0.8010 |
| **F1-score** | 0.7393 | 0.7426 | 0.7611 | **0.7645** | 0.7479 |
| **AUC** | 0.7945 | 0.8010 | **0.8219** | 0.7992 | 0.7941 |

SVM achieved the highest F1-score (0.7645) due to its superior Recall (0.8579), while XGBoost achieved the highest AUC (0.8219) and Accuracy (0.7389), demonstrating its stronger generalization capability across the decision boundary (Fig. 4).

**Fig. 4: Churn Prediction - Model Comparison, ROC Curves, Confusion Matrix, and Feature Importance**

![Churn Prediction Charts](data/generated/churn_prediction_charts.png)

### C. Hybrid Recommendation Engine Optimization
A weighted hybrid engine (60% Collaborative Filtering + 40% Content-Based CNN/Sentiment) was built to solve the cold-start problem (Fig. 5).

**Fig. 5: Hybrid Recommendation Engine - Content Scores, User Similarity Distribution, System Architecture, and Rating vs Sentiment Correlation**

![Recommendation Engine Charts](data/generated/recommendation_engine_charts.png)

## V. LIST OF ALL FIGURES

| Figure | Description | Source File |
| :---: | :--- | :--- |
| **Fig. 1** | CNN Confusion Matrix (15 clothing classes) | `cnn_confusion_matrix.png` |
| **Fig. 2** | Association Rules (Support vs Confidence vs Lift) | `association_rules_chart.png` |
| **Fig. 3** | Sentiment Analysis (Rating, Category, Polarity) | `sentiment_analysis_charts.png` |
| **Fig. 4** | Churn Prediction (ROC, Confusion Matrix, Features) | `churn_prediction_charts.png` |
| **Fig. 5** | Recommendation Engine (Architecture + Analysis) | `recommendation_engine_charts.png` |

## VI. CONCLUSION AND FUTURE WORK

A comprehensive machine learning-based data warehouse was developed for e-commerce environments. By integrating standard tabular modeling (Random Forest) with deep learning vision models (CNN) and natural language processing (VADER/TextBlob), the system provides multidimensional insights into customer behavior. Future work could explore the deployment of real-time streaming architectures (e.g., Apache Kafka) to update recommendation matrices instantaneously upon user interaction.
