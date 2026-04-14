"""
Step 4: Export Final Dataset from MySQL to CSV
Run this AFTER executing SQL scripts in phpMyAdmin.
The exported CSV is the ONLY file used for ML in Python.

Run: python scripts/03_export_final.py
"""
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text

# Config
DB_USER = 'root'
DB_PASS = 'root'
DB_HOST = 'localhost'
DB_PORT = 3306
DB_NAME = 'ecommerce_dm'

BASE = Path(__file__).resolve().parent.parent
DATA_GENERATED = BASE / 'data' / 'generated'
DATA_PROCESSED = BASE / 'data' / 'processed'
DATA_GENERATED.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

print('=' * 60)
print('  Step 4: Export Final Dataset from MySQL')
print('=' * 60)

engine = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# ============================================================
# 1. Export final_customer_dataset (THE generated dataset)
# ============================================================
print('\n📤 Exporting final_customer_dataset...')
df_final = pd.read_sql('SELECT * FROM final_customer_dataset', engine)
df_final.to_csv(DATA_GENERATED / 'customer_features.csv', index=False)
print(f'  ✅ {df_final.shape[0]} rows × {df_final.shape[1]} columns')
print(f'  → data/generated/customer_features.csv')

# ============================================================
# 2. Export clean tables (for reference)
# ============================================================
print('\n📤 Exporting clean tables...')

df_retail = pd.read_sql('SELECT * FROM clean_retail', engine)
df_retail.to_csv(DATA_PROCESSED / 'retail_clean.csv', index=False)
print(f'  ✅ clean_retail: {len(df_retail):,} rows → data/processed/retail_clean.csv')

df_reviews = pd.read_sql('SELECT * FROM clean_reviews', engine)
df_reviews.to_csv(DATA_PROCESSED / 'reviews_clean.csv', index=False)
print(f'  ✅ clean_reviews: {len(df_reviews):,} rows → data/processed/reviews_clean.csv')

df_amazon = pd.read_sql('SELECT * FROM clean_amazon', engine)
df_amazon.to_csv(DATA_PROCESSED / 'amazon_clean.csv', index=False)
print(f'  ✅ clean_amazon: {len(df_amazon):,} rows → data/processed/amazon_clean.csv')

# ============================================================
# 3. Print Data Warehouse Summary
# ============================================================
print('\n' + '=' * 60)
print('  DATA WAREHOUSE SUMMARY')
print('=' * 60)

tables = ['Dim_Customer', 'Dim_Product', 'Dim_Time', 'Dim_Location',
          'Fact_Orders', 'Fact_Reviews', 'final_customer_dataset']
with engine.connect() as conn:
    for table in tables:
        try:
            count = conn.execute(text(f'SELECT COUNT(*) FROM {table}')).scalar()
            print(f'  📋 {table}: {count:,} rows')
        except:
            print(f'  ❌ {table}: not found')

print('\n📊 Final Dataset Preview:')
print(df_final.head().to_string())
print(f'\nColumns: {list(df_final.columns)}')
print(f'\nChurn Distribution:')
print(df_final['ChurnLabel'].value_counts().to_string())

print('\n✅ Export Complete!')
print('👉 Now use data/generated/customer_features.csv for ML in Python')
