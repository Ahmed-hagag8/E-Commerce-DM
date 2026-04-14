"""
Day 9: Churn Prediction - Supervised Machine Learning
Trains and compares 5 ML models to predict customer churn.
Models: Logistic Regression, Random Forest, XGBoost, SVM, MLP Neural Network

Run: python scripts/07_churn_prediction.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import time

warnings.filterwarnings('ignore')

print('=' * 60)
print('  Day 9: Churn Prediction (Supervised ML)')
print('=' * 60)

# ============================================================
# CONFIG
# ============================================================
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / 'data' / 'generated'
OUTPUT_DIR = BASE / 'data' / 'generated'
MODEL_DIR = BASE / 'models'
MODEL_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. Load the Final Customer Dataset
# ============================================================
print('\n1. Loading customer features dataset...')

df = pd.read_csv(DATA_DIR / 'customer_features.csv')
print(f'   Shape: {df.shape[0]:,} customers x {df.shape[1]} features')
print(f'   Columns: {list(df.columns)}')

# ============================================================
# 2. Data Preparation
# ============================================================
print('\n2. Preparing data for ML...')

# Drop rows with missing target
df = df.dropna(subset=['ChurnLabel'])

# Show class distribution
churn_dist = df['ChurnLabel'].value_counts()
print(f'   Churn distribution:')
print(f'     Active (0): {churn_dist.get(0, 0):,}')
print(f'     Churned (1): {churn_dist.get(1, 0):,}')
print(f'     Churn Rate: {churn_dist.get(1, 0) / len(df) * 100:.1f}%')

# Select features (drop identifiers, dates, and target)
drop_cols = ['CustomerID', 'JoinDate', 'FirstOrderDate', 'LastOrderDate',
             'ChurnLabel', 'Country']
feature_cols = [c for c in df.columns if c not in drop_cols]

# Encode categorical columns
le = LabelEncoder()
if 'Region' in feature_cols:
    df['Region'] = le.fit_transform(df['Region'].fillna('Unknown'))

# Fill remaining NaN with 0
df[feature_cols] = df[feature_cols].fillna(0)

X = df[feature_cols].values
y = df['ChurnLabel'].astype(int).values

print(f'   Features used ({len(feature_cols)}): {feature_cols}')

# Train/Test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (important for SVM and MLP)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f'   Train set: {len(X_train):,} | Test set: {len(X_test):,}')

# ============================================================
# 3. Define and Train Models
# ============================================================
print('\n3. Training 5 ML models...')

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, random_state=42,
                              use_label_encoder=False, eval_metric='logloss'),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'MLP Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                         random_state=42, early_stopping=True)
}

results = {}

for name, model in models.items():
    start = time.time()
    print(f'\n   Training {name}...')

    # SVM and MLP need scaled data
    if name in ['SVM', 'MLP Neural Network', 'Logistic Regression']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    elapsed = time.time() - start

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'time': elapsed
    }

    print(f'     Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Time: {elapsed:.1f}s')

# ============================================================
# 4. Model Comparison Table
# ============================================================
print('\n' + '=' * 60)
print('  MODEL COMPARISON')
print('=' * 60)
print(f'  {"Model":<25} {"Accuracy":>9} {"Precision":>10} {"Recall":>8} {"F1":>8} {"AUC":>8}')
print('  ' + '-' * 69)

best_model_name = None
best_f1 = 0

for name, r in results.items():
    print(f'  {name:<25} {r["accuracy"]:>9.4f} {r["precision"]:>10.4f} {r["recall"]:>8.4f} {r["f1"]:>8.4f} {r["auc"]:>8.4f}')
    if r['f1'] > best_f1:
        best_f1 = r['f1']
        best_model_name = name

print(f'\n  BEST MODEL: {best_model_name} (F1 = {best_f1:.4f})')

# ============================================================
# 5. Detailed Report for Best Model
# ============================================================
print(f'\n  Classification Report ({best_model_name}):')
print(classification_report(y_test, results[best_model_name]['y_pred'],
                            target_names=['Active', 'Churned']))

# ============================================================
# 6. Visualizations
# ============================================================
print('\n4. Generating visualizations...')

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot 1: Model Comparison Bar Chart
model_names = list(results.keys())
metrics_data = {
    'Accuracy': [results[m]['accuracy'] for m in model_names],
    'Precision': [results[m]['precision'] for m in model_names],
    'Recall': [results[m]['recall'] for m in model_names],
    'F1 Score': [results[m]['f1'] for m in model_names],
    'AUC': [results[m]['auc'] for m in model_names]
}

x = np.arange(len(model_names))
width = 0.15
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

for i, (metric, values) in enumerate(metrics_data.items()):
    axes[0, 0].bar(x + i * width, values, width, label=metric, color=colors[i], edgecolor='black', linewidth=0.3)

axes[0, 0].set_xticks(x + width * 2)
axes[0, 0].set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=8)
axes[0, 0].set_ylabel('Score')
axes[0, 0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[0, 0].legend(loc='lower right', fontsize=8)
axes[0, 0].set_ylim(0, 1.1)

# Plot 2: ROC Curves
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
    axes[0, 1].plot(fpr, tpr, label=f'{name} (AUC={r["auc"]:.3f})', linewidth=2)

axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curves', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=8)

# Plot 3: Confusion Matrix for Best Model
cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['Active', 'Churned'], yticklabels=['Active', 'Churned'])
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_title(f'Confusion Matrix ({best_model_name})', fontsize=14, fontweight='bold')

# Plot 4: Feature Importance (from Random Forest)
rf_model = results['Random Forest']['model']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:10]  # Top 10

bar_colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
axes[1, 1].barh(range(len(indices)), importances[indices][::-1],
                color=bar_colors, edgecolor='black', linewidth=0.5)
axes[1, 1].set_yticks(range(len(indices)))
axes[1, 1].set_yticklabels([feature_cols[i] for i in indices][::-1], fontsize=9)
axes[1, 1].set_xlabel('Importance')
axes[1, 1].set_title('Top 10 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')

plt.suptitle('Churn Prediction - Model Evaluation', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

chart_path = os.path.join(str(OUTPUT_DIR), 'churn_prediction_charts.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f'   Charts saved to: {chart_path}')

# Save comparison table
comparison_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': [results[m]['accuracy'] for m in model_names],
    'Precision': [results[m]['precision'] for m in model_names],
    'Recall': [results[m]['recall'] for m in model_names],
    'F1_Score': [results[m]['f1'] for m in model_names],
    'AUC': [results[m]['auc'] for m in model_names],
    'Training_Time_Seconds': [results[m]['time'] for m in model_names]
})
comparison_path = os.path.join(str(OUTPUT_DIR), 'churn_model_comparison.csv')
comparison_df.to_csv(comparison_path, index=False)
print(f'   Comparison saved to: {comparison_path}')

# ============================================================
# 7. Summary
# ============================================================
print('\n' + '=' * 60)
print('  CHURN PREDICTION SUMMARY')
print('=' * 60)
print(f'  Total customers: {len(df):,}')
print(f'  Features used: {len(feature_cols)}')
print(f'  Models trained: {len(models)}')
print(f'  Best model: {best_model_name}')
print(f'  Best F1 Score: {best_f1:.4f}')
print(f'  Best AUC: {results[best_model_name]["auc"]:.4f}')

print('\n' + '=' * 60)
print('  DAY 9 COMPLETED SUCCESSFULLY!')
print('=' * 60)
print('  Next: python scripts/08_recommendation.py')
