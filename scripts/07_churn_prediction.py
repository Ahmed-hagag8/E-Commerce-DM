import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, StackingClassifier
from imblearn.over_sampling import SMOTE
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

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / 'data' / 'generated'
OUTPUT_DIR = BASE / 'data' / 'generated'
MODEL_DIR = BASE / 'models'
MODEL_DIR.mkdir(exist_ok=True)

DB_USER = 'root'
DB_PASS = 'root'
DB_HOST = 'localhost'
DB_PORT = 3306
DB_NAME = 'ecommerce_dm'

print('\n1. Loading customer features and enriching from database...')

df = pd.read_csv(DATA_DIR / 'customer_features.csv')
print(f'   Base shape: {df.shape[0]:,} customers x {df.shape[1]} features')

from sqlalchemy import create_engine, text as sa_text
engine = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

print('   Extracting behavioral features from Fact_Orders...')
extra_features = pd.read_sql("""
    SELECT 
        fo.CustomerID,
        COUNT(DISTINCT dp.Category) AS CategoryDiversity,
        COUNT(DISTINCT CASE WHEN dt.IsWeekend = 1 THEN fo.OrderID END) * 1.0 
            / NULLIF(COUNT(fo.OrderID), 0) AS WeekendRatio,
        SUM(CASE WHEN dp.PriceRange = 'Premium' THEN 1 ELSE 0 END) * 1.0 
            / NULLIF(COUNT(fo.OrderID), 0) AS PremiumRatio,
        MAX(dt.Month) AS LastPurchaseMonth,
        STDDEV(fo.TotalPrice) AS SpendingStdDev
    FROM Fact_Orders fo
    JOIN Dim_Product dp ON fo.ProductID = dp.ProductID
    JOIN Dim_Time dt ON fo.TimeID = dt.TimeID
    GROUP BY fo.CustomerID
""", engine)

df = df.merge(extra_features, on='CustomerID', how='left')
df['CategoryDiversity'] = df['CategoryDiversity'].fillna(0)
df['WeekendRatio'] = df['WeekendRatio'].fillna(0)
df['PremiumRatio'] = df['PremiumRatio'].fillna(0)
df['LastPurchaseMonth'] = df['LastPurchaseMonth'].fillna(0)
df['SpendingStdDev'] = df['SpendingStdDev'].fillna(0)

print(f'   Enriched shape: {df.shape[0]:,} customers x {df.shape[1]} features')
print(f'   New features: CategoryDiversity, WeekendRatio, PremiumRatio, LastPurchaseMonth, SpendingStdDev')

print('\n2. Preparing data for ML...')

df = df.dropna(subset=['ChurnLabel'])

df['MonetaryPerProduct'] = df['Monetary'] / (df['UniqueProducts'] + 1)
df['HighValue'] = (df['Monetary'] > df['Monetary'].median()).astype(int)
df['IsOneTimeBuyer'] = (df['Frequency'] == 1).astype(int)
df['LowSpender'] = (df['Monetary'] < df['Monetary'].quantile(0.25)).astype(int)

df['OrdersPerLifespan'] = df['Frequency'] / (df['CustomerLifespanDays'] + 1)
df['AvgDaysBetweenOrders'] = (df['CustomerLifespanDays'] + 1) / (df['Frequency'] + 1)
df['MonetaryPerLifespan'] = df['Monetary'] / (df['CustomerLifespanDays'] + 1)
df['LogLifespan'] = np.log1p(df['CustomerLifespanDays'])

df['LogMonetary'] = np.log1p(df['Monetary'])
df['LogFrequency'] = np.log1p(df['Frequency'])
df['MonetaryFrequencyRatio'] = df['Monetary'] / (df['Frequency'] + 1)
df['FrequencyPerProduct'] = df['Frequency'] / (df['UniqueProducts'] + 1)
df['FrequencySquared'] = df['Frequency'] ** 2
df['HighFrequencyHighValue'] = ((df['Frequency'] > df['Frequency'].median()) & (df['Monetary'] > df['Monetary'].median())).astype(int)
df['SpendingConsistency'] = df['SpendingStdDev'] / (df['Monetary'] / (df['Frequency'] + 1) + 1)

outlier_cols = ['Monetary', 'Frequency', 'UniqueProducts', 'AvgPricePerItem']
for col in outlier_cols:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    before = len(df[(df[col] < lower) | (df[col] > upper)])
    df[col] = df[col].clip(lower, upper)
    if before > 0:
        print(f'   Winsorized {col}: {before} outliers capped')

churn_dist = df['ChurnLabel'].value_counts()
print(f'   Churn distribution:')
print(f'     Active (0): {churn_dist.get(0, 0):,}')
print(f'     Churned (1): {churn_dist.get(1, 0):,}')
print(f'     Churn Rate: {churn_dist.get(1, 0) / len(df) * 100:.1f}%')

drop_cols = ['CustomerID', 'JoinDate', 'FirstOrderDate', 'LastOrderDate',
             'ChurnLabel', 'Country', 'Recency',
             'TotalOrders', 'AvgOrderValue', 'AvgQuantityPerOrder', 'TotalItems']
feature_cols = [c for c in df.columns if c not in drop_cols]

le = LabelEncoder()
if 'Region' in feature_cols:
    df['Region'] = le.fit_transform(df['Region'].fillna('Unknown'))

df[feature_cols] = df[feature_cols].fillna(0)

X = df[feature_cols].values
y = df['ChurnLabel'].astype(int).values

print(f'   Features used ({len(feature_cols)}): {feature_cols}')

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
)

print('\n   Class balance is near 50/50 — skipping SMOTE, using class_weight instead...')
print(f'   Train: Active={sum(y_train==0):,}, Churned={sum(y_train==1):,}')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f'   Train set: {len(X_train):,} | Validation set: {len(X_val):,} | Test set: {len(X_test):,}')

print('\n3. Hyperparameter tuning & training 5 ML models...')

print('\n   🔧 Tuning Random Forest...')
rf_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [8, 12, 16, 20, 25, None],
    'min_samples_split': [2, 3, 5, 8],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt', 'log2', 0.5, 0.7]
}
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
    rf_params, n_iter=30, cv=5, scoring='accuracy', random_state=42, n_jobs=-1
)
rf_search.fit(X_train_scaled, y_train)
print(f'     Best RF params: {rf_search.best_params_}')
print(f'     Best RF CV Accuracy: {rf_search.best_score_:.4f}')

print('\n   🔧 Tuning XGBoost...')
xgb_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [1.0, 1.5, 2.0],
    'min_child_weight': [1, 3, 5]
}
xgb_search = RandomizedSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss'),
    xgb_params, n_iter=30, cv=5, scoring='accuracy', random_state=42, n_jobs=-1
)
xgb_search.fit(X_train_scaled, y_train)
print(f'     Best XGB params: {xgb_search.best_params_}')
print(f'     Best XGB CV Accuracy: {xgb_search.best_score_:.4f}')

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=1.0, class_weight='balanced'),
    'Random Forest': rf_search.best_estimator_,
    'XGBoost': xgb_search.best_estimator_,
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=500, max_depth=4, learning_rate=0.03,
                                                     subsample=0.8, min_samples_leaf=5, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42, C=2.0, gamma='scale', class_weight='balanced'),
    'MLP Neural Network': MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=800,
                                         random_state=42, early_stopping=True,
                                         learning_rate='adaptive', alpha=0.001)
}

results = {}

for name, model in models.items():
    start = time.time()
    print(f'\n   Training {name}...')

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_proba_val = model.predict_proba(X_val_scaled)[:, 1]

    elapsed = time.time() - start

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'y_proba_val': y_proba_val,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'time': elapsed
    }

    print(f'     Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Time: {elapsed:.1f}s')

print('\n   Training Weighted Voting Ensemble (All 6 Models)...')
start = time.time()

weights = {'Gradient Boosting': 2.0, 'XGBoost': 2.0, 'MLP Neural Network': 1.5,
           'Random Forest': 1.0, 'SVM': 1.0, 'Logistic Regression': 0.5}
total_w = sum(weights.values())

y_proba_ensemble = sum(results[m]['y_proba'] * w for m, w in weights.items()) / total_w
y_pred_ensemble = (y_proba_ensemble >= 0.5).astype(int)
y_proba_ensemble_val = sum(results[m]['y_proba_val'] * w for m, w in weights.items()) / total_w

elapsed = time.time() - start
acc = accuracy_score(y_test, y_pred_ensemble)
prec = precision_score(y_test, y_pred_ensemble)
rec = recall_score(y_test, y_pred_ensemble)
f1 = f1_score(y_test, y_pred_ensemble)
auc = roc_auc_score(y_test, y_proba_ensemble)

results['Voting Ensemble'] = {
    'model': None,
    'y_pred': y_pred_ensemble,
    'y_proba': y_proba_ensemble,
    'y_proba_val': y_proba_ensemble_val,
    'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc,
    'time': elapsed
}
print(f'     Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Time: {elapsed:.1f}s')

print('\n   Training Stacking Ensemble (All Models -> Meta: Logistic Regression)...')
start = time.time()
estimators = [
    ('xgb', results['XGBoost']['model']),
    ('rf', results['Random Forest']['model']),
    ('mlp', results['MLP Neural Network']['model']),
    ('gb', results['Gradient Boosting']['model']),
    ('svm', results['SVM']['model'])
]
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(random_state=42, C=1.0, max_iter=1000),
    cv=5,
    n_jobs=-1
)
stacking.fit(X_train_scaled, y_train)
y_pred_stack = stacking.predict(X_test_scaled)
y_proba_stack = stacking.predict_proba(X_test_scaled)[:, 1]
y_proba_stack_val = stacking.predict_proba(X_val_scaled)[:, 1]
elapsed_stack = time.time() - start

acc = accuracy_score(y_test, y_pred_stack)
prec = precision_score(y_test, y_pred_stack)
rec = recall_score(y_test, y_pred_stack)
f1 = f1_score(y_test, y_pred_stack)
auc = roc_auc_score(y_test, y_proba_stack)
results['Stacking Ensemble'] = {
    'model': stacking,
    'y_pred': y_pred_stack,
    'y_proba': y_proba_stack,
    'y_proba_val': y_proba_stack_val,
    'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc,
    'time': elapsed_stack
}
print(f'     Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Time: {elapsed_stack:.1f}s')

print('\n   Tuning decision thresholds on validation set to maximize Accuracy...')
for name in list(results.keys()):
    best_acc_thr = 0
    best_thr = 0.5
    y_p_val = results[name]['y_proba_val']
    for thr in np.arange(0.30, 0.70, 0.005):
        y_pred_thr = (y_p_val >= thr).astype(int)
        acc_thr = accuracy_score(y_val, y_pred_thr)
        if acc_thr > best_acc_thr:
            best_acc_thr = acc_thr
            best_thr = thr

    y_pred_test = (results[name]['y_proba'] >= best_thr).astype(int)
    results[name]['y_pred'] = y_pred_test
    results[name]['accuracy'] = accuracy_score(y_test, y_pred_test)
    results[name]['precision'] = precision_score(y_test, y_pred_test)
    results[name]['recall'] = recall_score(y_test, y_pred_test)
    results[name]['f1'] = f1_score(y_test, y_pred_test)
    results[name]['threshold'] = best_thr

print('\n' + '=' * 60)
print('  MODEL COMPARISON (With Optimized F1 Thresholds)')
print('=' * 60)
print(f'  {"Model":<25} {"Thr":<5} {"Accuracy":>9} {"Precision":>10} {"Recall":>8} {"F1":>8} {"AUC":>8}')
print('  ' + '-' * 75)

best_model_name = None
best_f1 = 0

for name, r in results.items():
    thr_str = f"{r.get('threshold', 0.5):.2f}"
    print(f'  {name:<25} {thr_str:<5} {r["accuracy"]:>9.4f} {r["precision"]:>10.4f} {r["recall"]:>8.4f} {r["f1"]:>8.4f} {r["auc"]:>8.4f}')
    if r['f1'] > best_f1:
        best_f1 = r['f1']
        best_model_name = name

print(f'\n  BEST MODEL: {best_model_name} (F1 = {best_f1:.4f})')

print(f'\n  Classification Report ({best_model_name}):')
print(classification_report(y_test, results[best_model_name]['y_pred'],
                             target_names=['Active', 'Churned']))

print('\n4. Generating visualizations...')

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

axes[0, 0].axis('off')
axes[0, 0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')

model_names = list(results.keys())
table_data = []
for name in model_names:
    r = results[name]
    thr_str = f"{r.get('threshold', 0.5):.2f}"
    short_name = name.replace(' Regression', '\nRegression').replace(' Network', '\nNetwork').replace(' Ensemble', '\nEnsemble')
    table_data.append([
        short_name,
        thr_str,
        f"{r['accuracy']:.3f}",
        f"{r['precision']:.3f}",
        f"{r['recall']:.3f}",
        f"{r['f1']:.3f}",
        f"{r['auc']:.3f}"
    ])

cols = ['Model', 'Thr', 'Acc', 'Prec', 'Recall', 'F1', 'AUC']
colWidths = [0.26, 0.11, 0.12, 0.12, 0.12, 0.12, 0.12]

table = axes[0, 0].table(cellText=table_data,
                         colLabels=cols,
                         colWidths=colWidths,
                         cellLoc='center',
                         loc='center',
                         bbox=[0.05, 0.1, 0.95, 0.8])

table.auto_set_font_size(False)
table.set_fontsize(11)

for (row, col), cell in table.get_celld().items():
    cell.set_height(0.12)
    if row == 0:
        cell.set_text_props(weight='bold', color='white', fontsize=12)
        cell.set_facecolor('#2c3e50')
    elif row % 2 == 0:
        cell.set_facecolor('#fefefe')
    else:
        cell.set_facecolor('#f4f6f7')

for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
    axes[0, 1].plot(fpr, tpr, label=f'{name} (AUC={r["auc"]:.3f})', linewidth=2)

axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curves', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=8)

cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['Active', 'Churned'], yticklabels=['Active', 'Churned'])
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_title(f'Confusion Matrix ({best_model_name})', fontsize=14, fontweight='bold')

rf_model = results['Random Forest']['model']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:10]

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
