import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

print('=' * 60)
print('  Day 8: Association Rules (Market Basket Analysis)')
print('=' * 60)

DB_USER = 'root'
DB_PASS = 'root'
DB_HOST = 'localhost'
DB_PORT = 3306
DB_NAME = 'ecommerce_dm'

OUTPUT_DIR = r'E:\Projects\E-Commerce DM\data\generated'
os.makedirs(OUTPUT_DIR, exist_ok=True)

engine = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

print('\n1. Loading transaction data from Fact_Orders...')

df = pd.read_sql("""
    SELECT fo.InvoiceNo, dp.ProductName
    FROM Fact_Orders fo
    JOIN Dim_Product dp ON fo.ProductID = dp.ProductID
    WHERE fo.Quantity > 0
      AND fo.InvoiceNo IN (
          SELECT InvoiceNo FROM Fact_Orders
          GROUP BY InvoiceNo
          HAVING COUNT(DISTINCT ProductID) >= 2
      )
""", engine)

print(f'   Total transaction rows: {len(df):,}')
print(f'   Unique invoices (multi-item only): {df["InvoiceNo"].nunique():,}')
print(f'   Unique products: {df["ProductName"].nunique():,}')

print('\n2. Filtering to top 100 most frequently purchased products...')

top_products = df['ProductName'].value_counts().head(100).index.tolist()
df_filtered = df[df['ProductName'].isin(top_products)]

print(f'   Filtered transaction rows: {len(df_filtered):,}')
print(f'   Unique invoices (filtered): {df_filtered["InvoiceNo"].nunique():,}')

print('\n3. Creating basket matrix (Invoice x Product)...')

basket = df_filtered.groupby(['InvoiceNo', 'ProductName'])['ProductName'].count().unstack().fillna(0)

basket = (basket > 0).astype(int)

print(f'   Basket shape: {basket.shape[0]:,} invoices x {basket.shape[1]} products')

print('\n4. Running Apriori algorithm (min_support=0.03)...')

frequent_itemsets = apriori(basket, min_support=0.03, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)

print(f'   Found {len(frequent_itemsets)} frequent itemsets')
print(f'   - Single items: {len(frequent_itemsets[frequent_itemsets["length"] == 1])}')
print(f'   - Pairs: {len(frequent_itemsets[frequent_itemsets["length"] == 2])}')
print(f'   - Triples: {len(frequent_itemsets[frequent_itemsets["length"] >= 3])}')

print(f'   Support range: {frequent_itemsets["support"].min():.4f} - {frequent_itemsets["support"].max():.4f}')
print(f'   Median support: {frequent_itemsets["support"].median():.4f}')
print(f'   min_support=0.03 means an itemset must appear in >= {int(0.03 * len(basket)):,} of {len(basket):,} invoices')

print('\n5. Generating association rules (min_confidence=0.3)...')

rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.3,
                          num_itemsets=len(basket))

rules = rules.sort_values('lift', ascending=False)

rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

print(f'   Found {len(rules)} association rules')

print('\n' + '=' * 60)
print('  TOP 15 ASSOCIATION RULES (sorted by Lift)')
print('=' * 60)

top_rules = rules.head(15)[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']]
top_rules.columns = ['IF Customer Buys', 'THEN They Also Buy', 'Support', 'Confidence', 'Lift']

for i, row in top_rules.iterrows():
    print(f'\n  Rule: IF [{row["IF Customer Buys"]}]')
    print(f'        THEN [{row["THEN They Also Buy"]}]')
    print(f'        Support: {row["Support"]:.3f} | Confidence: {row["Confidence"]:.2%} | Lift: {row["Lift"]:.2f}')

print('\n6. Generating visualizations...')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

scatter = axes[0].scatter(
    rules['support'], rules['confidence'],
    c=rules['lift'], cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=0.5, s=60
)
axes[0].set_xlabel('Support', fontsize=12)
axes[0].set_ylabel('Confidence', fontsize=12)
axes[0].set_title('Association Rules: Support vs Confidence', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=axes[0], label='Lift')

top10 = rules.head(10).copy()
top10['rule_label'] = top10['antecedents_str'].str[:25] + ' → ' + top10['consequents_str'].str[:25]
top10 = top10.iloc[::-1]

colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top10)))
axes[1].barh(range(len(top10)), top10['lift'], color=colors, edgecolor='black', linewidth=0.5)
axes[1].set_yticks(range(len(top10)))
axes[1].set_yticklabels(top10['rule_label'], fontsize=9)
axes[1].set_xlabel('Lift', fontsize=12)
axes[1].set_title('Top 10 Rules by Lift', fontsize=14, fontweight='bold')

plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, 'association_rules_chart.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f'   Chart saved to: {chart_path}')

rules_export = rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].copy()
rules_export.columns = ['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift']
rules_path = os.path.join(OUTPUT_DIR, 'association_rules.csv')
rules_export.to_csv(rules_path, index=False)
print(f'   Rules saved to: {rules_path}')

print('\n' + '=' * 60)
print('  BUSINESS INSIGHTS')
print('=' * 60)

high_confidence = rules[rules['confidence'] >= 0.5]
high_lift = rules[rules['lift'] >= 3]

print(f'  Total rules discovered: {len(rules)}')
print(f'  High-confidence rules (>=50%): {len(high_confidence)}')
print(f'  Strong association rules (lift >= 3): {len(high_lift)}')

if len(high_lift) > 0:
    best = high_lift.iloc[0]
    print(f'\n  STRONGEST RULE:')
    print(f'  Customers who buy [{best["antecedents_str"]}]')
    print(f'  are {best["lift"]:.1f}x MORE LIKELY to also buy [{best["consequents_str"]}]')
    print(f'  (Confidence: {best["confidence"]:.1%})')

print('\n' + '=' * 60)
print('  DAY 8 COMPLETED SUCCESSFULLY!')
print('=' * 60)
print('  Next: python scripts/06_sentiment_analysis.py')
