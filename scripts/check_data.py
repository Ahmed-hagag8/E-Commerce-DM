import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('mysql+pymysql://root:root@localhost:3306/ecommerce_dm')

tables = pd.read_sql('SHOW TABLES', engine)
print('=== TABLES ===')
print(tables.to_string())

print('\n=== Fact_Orders Columns ===')
cols = pd.read_sql('SELECT * FROM Fact_Orders LIMIT 2', engine)
print(cols.columns.tolist())
print(f'Rows: {pd.read_sql("SELECT COUNT(*) as c FROM Fact_Orders", engine).iloc[0,0]}')

print('\n=== Dim_Customer Columns ===')
cols = pd.read_sql('SELECT * FROM Dim_Customer LIMIT 2', engine)
print(cols.columns.tolist())

print('\n=== Dim_Product Columns ===')
cols = pd.read_sql('SELECT * FROM Dim_Product LIMIT 2', engine)
print(cols.columns.tolist())

print('\n=== Fact_Reviews Columns ===')
cols = pd.read_sql('SELECT * FROM Fact_Reviews LIMIT 2', engine)
print(cols.columns.tolist())
print(f'Rows: {pd.read_sql("SELECT COUNT(*) as c FROM Fact_Reviews", engine).iloc[0,0]}')

print('\n=== Dim_Time Columns ===')
cols = pd.read_sql('SELECT * FROM Dim_Time LIMIT 2', engine)
print(cols.columns.tolist())

print('\n=== Dim_Location Columns ===')
try:
    cols = pd.read_sql('SELECT * FROM Dim_Location LIMIT 2', engine)
    print(cols.columns.tolist())
except:
    print('Table not found')

print('\n=== Sample Reviews ===')
print(pd.read_sql('SELECT * FROM Fact_Reviews LIMIT 3', engine).to_string())

print('\n=== Sample Products ===')
print(pd.read_sql('SELECT * FROM Dim_Product LIMIT 5', engine).to_string())

print('\n=== Sample Orders with Time ===')
print(pd.read_sql('''
    SELECT fo.OrderID, fo.Quantity, fo.UnitPrice, fo.TotalPrice, 
           dt.FullDate, dt.Year, dt.Month, dt.IsWeekend,
           dp.Category, dp.PriceRange
    FROM Fact_Orders fo
    JOIN Dim_Time dt ON fo.TimeID = dt.TimeID
    JOIN Dim_Product dp ON fo.ProductID = dp.ProductID
    LIMIT 5
''', engine).to_string())
