"""
Run ALL SQL scripts via PyMySQL multi-statement mode
Run: python scripts/run_all_sql.py
"""
import pymysql
from pymysql.constants import CLIENT
import time
from pathlib import Path

DB_USER = 'root'
DB_PASS = 'root'
DB_HOST = 'localhost'
DB_PORT = 3306
DB_NAME = 'ecommerce_dm'

SQL_DIR = Path(__file__).resolve().parent.parent / 'sql'

sql_files = [
    '02a_clean_retail.sql',
    '02b_clean_reviews_amazon.sql',
    '03a_dimensions.sql',
    '03b_rfm_customer.sql',
]

print('=' * 60)
print('  Running ALL SQL Preprocessing')
print('=' * 60)

for f in sql_files:
    filepath = SQL_DIR / f
    print(f'\n⏳ Running: {f}...')
    start = time.time()
    
    sql = filepath.read_text(encoding='utf-8')
    
    # Connect with MULTI_STATEMENTS flag — runs entire SQL file at once
    conn = pymysql.connect(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASS,
        database=DB_NAME,
        client_flag=CLIENT.MULTI_STATEMENTS,
        connect_timeout=600,
        read_timeout=600,
        write_timeout=600,
    )
    
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        
        # Fetch all result sets (from SELECT statements)
        results = []
        while True:
            try:
                rows = cursor.fetchall()
                if rows and cursor.description:
                    cols = [d[0] for d in cursor.description]
                    for row in rows:
                        results.append(dict(zip(cols, row)))
            except:
                pass
            if not cursor.nextset():
                break
        
        conn.commit()
        elapsed = time.time() - start
        print(f'  ✅ Done ({elapsed:.1f}s)')
        
        # Show results
        for r in results:
            print(f'  → {r}')
            
    except Exception as e:
        elapsed = time.time() - start
        print(f'  ❌ Error after {elapsed:.1f}s: {str(e)[:200]}')
    finally:
        conn.close()

print('\n' + '=' * 60)
print('  ALL SQL PREPROCESSING DONE!')
print('=' * 60)
print('👉 Next: python scripts/run_03c_fast.py')
