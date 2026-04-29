"""
Step 1: Load Raw Data into MySQL Staging Tables
NO cleaning here — just raw data loading into MySQL.
All preprocessing will happen in SQL.

Run: python scripts/01_load_raw.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.types import VARCHAR, INT, DECIMAL, DATETIME, FLOAT
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
DB_USER = 'root'
DB_PASS = 'root'
DB_HOST = 'localhost'
DB_PORT = 3306
DB_NAME = 'ecommerce_dm'

BASE = Path(__file__).resolve().parent.parent
DATA_RAW = BASE / 'data' / 'raw'

print('=' * 60)
print('  Step 1: Load Raw Data into MySQL Staging Tables')
print('=' * 60)

# ============================================================
# Connect to MySQL
# ============================================================
engine_base = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/')
with engine_base.connect() as conn:
    conn.execute(text(f'CREATE DATABASE IF NOT EXISTS {DB_NAME}'))
    conn.commit()

engine = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# Disable FK checks for clean slate
with engine.connect() as conn:
    conn.execute(text('SET FOREIGN_KEY_CHECKS = 0'))
    conn.commit()

print('✅ Connected to MySQL\n')

# ============================================================
# 1. Load Online Retail II → staging_retail
# ============================================================
print('📥 Loading Online Retail II (this takes ~2 min for 43MB xlsx)...')
retail_path = DATA_RAW / 'online_retail' / 'online_retail_II.xlsx'
df_y1 = pd.read_excel(retail_path, sheet_name='Year 2009-2010')
df_y2 = pd.read_excel(retail_path, sheet_name='Year 2010-2011')
df_retail = pd.concat([df_y1, df_y2], ignore_index=True)

# Convert types for MySQL compatibility (minimal — NOT cleaning)
df_retail['Invoice'] = df_retail['Invoice'].astype(str)
df_retail['InvoiceDate'] = pd.to_datetime(df_retail['InvoiceDate'])

df_retail.to_sql('staging_retail', engine, if_exists='replace', index=False,
                  dtype={
                      'Invoice': VARCHAR(20),
                      'StockCode': VARCHAR(20),
                      'Description': VARCHAR(255),
                      'Quantity': INT(),
                      'InvoiceDate': DATETIME(),
                      'Price': DECIMAL(10,2),
                      'Customer ID': FLOAT(),
                      'Country': VARCHAR(50)
                  })
print(f'  ✅ staging_retail: {len(df_retail):,} rows loaded')

# ============================================================
# 2. Load Reviews → staging_reviews
# ============================================================
print('\n📥 Loading Womens Clothing Reviews...')
reviews_path = DATA_RAW / 'clothing_reviews' / 'Womens Clothing E-Commerce Reviews.csv'
df_reviews = pd.read_csv(reviews_path)
df_reviews.to_sql('staging_reviews', engine, if_exists='replace', index=False)
print(f'  ✅ staging_reviews: {len(df_reviews):,} rows loaded')


# ============================================================
# Verify
# ============================================================
print('\n' + '=' * 60)
print('  STAGING TABLES LOADED')
print('=' * 60)
with engine.connect() as conn:
    for table in ['staging_retail', 'staging_reviews']:
        count = conn.execute(text(f'SELECT COUNT(*) FROM {table}')).scalar()
        print(f'  📋 {table}: {count:,} rows')

# Re-enable FK checks (were disabled at start for clean loading)
with engine.connect() as conn:
    conn.execute(text('SET FOREIGN_KEY_CHECKS = 1'))
    conn.commit()

print('\n✅ Step 1 Complete!')
print('👉 Next: Run the preprocessing pipeline:')
print('   1. python scripts/run_all_sql.py')
print('   2. python scripts/run_03c_fast.py')
print('   3. python scripts/03_export_final.py')
