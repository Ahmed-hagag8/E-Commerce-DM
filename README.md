# 🛒 E-Commerce Customer Behavior Analysis & Hybrid Recommendation System

> **AIE 323 – Data Mining** | New Mansoura University | Spring 2025/2026

## 📋 Overview

An end-to-end data mining system for e-commerce that analyzes customer behavior, predicts churn, discovers product associations, and delivers hybrid product recommendations.

## 🏗️ Architecture

```
Data Sources → ETL Pipeline → Data Warehouse (MySQL) → ML Models → Hybrid Recommender → Streamlit Dashboard
```

## 📊 Datasets

| Dataset | Type | Source |
|---------|------|--------|
| Online Retail II | Tabular | UCI ML Repository |
| Amazon Product Images | Image | Kaggle |
| Women's Clothing Reviews | Hybrid | Kaggle |

## 🧠 Models

- **K-Means Clustering** — Customer segmentation (RFM)
- **Apriori** — Market basket analysis
- **Logistic Regression / Random Forest / XGBoost / SVM / MLP** — Churn prediction
- **MobileNetV2 (CNN)** — Product image classification
- **VADER + TF-IDF** — Sentiment analysis
- **Hybrid Recommendation Engine** — 4-method weighted scoring

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/E-Commerce-DM.git
cd E-Commerce-DM

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download datasets (requires Kaggle API key)
python scripts/download_data.py

# 5. Run the Streamlit dashboard
streamlit run dashboards/streamlit_app.py
```

## 📁 Project Structure

```
E-Commerce-DM/
├── data/
│   ├── raw/              # Original datasets
│   ├── processed/        # Cleaned datasets
│   └── generated/        # Post-ETL merged dataset
├── sql/
│   ├── create_schema.sql # Star schema DDL
│   ├── etl_load.sql      # Data loading scripts
│   └── queries.sql       # Analytical queries
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_ETL.ipynb
│   ├── 03_Clustering.ipynb
│   ├── 04_Association.ipynb
│   ├── 05_Churn_ML.ipynb
│   ├── 06_Image_CNN.ipynb
│   ├── 07_Sentiment.ipynb
│   └── 08_Recommendation.ipynb
├── src/
│   ├── etl.py
│   ├── models.py
│   ├── recommender.py
│   └── utils.py
├── dashboards/
│   └── streamlit_app.py
├── models/               # Saved model files
├── results/
│   ├── figures/          # Generated charts
│   └── model_metrics.csv
└── scripts/
    └── download_data.py
```

## 📈 Key Results

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | — | — | — |
| Random Forest | — | — | — |
| XGBoost | — | — | — |
| SVM | — | — | — |
| MLP | — | — | — |

*Results will be populated after model training.*

## 👥 Team

- [Team member names here]

## 📄 License

This project is developed for academic purposes as part of AIE 323 – Data Mining course.
