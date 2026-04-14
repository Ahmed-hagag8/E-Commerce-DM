"""
Step 3C — FAST version with index optimization
Run: python scripts/run_03c_fast.py
"""
from sqlalchemy import create_engine, text
import time

engine = create_engine('mysql+pymysql://root:root@localhost:3306/ecommerce_dm')

print('='*60)
print('  Step 3C — FAST version')
print('='*60)

# Step 0: Add a DATE index to clean_retail for fast JOINs
print('\n⏳ Adding date index to clean_retail...')
with engine.connect() as conn:
    try:
        conn.execute(text('ALTER TABLE clean_retail ADD COLUMN InvoiceDay DATE'))
        conn.execute(text('UPDATE clean_retail SET InvoiceDay = DATE(InvoiceDate)'))
        conn.execute(text('CREATE INDEX idx_invoice_day ON clean_retail(InvoiceDay)'))
        conn.commit()
        print('  ✅ Date index added')
    except:
        conn.rollback()
        print('  ✅ Index already exists')

# Also add index on Dim_Time.FullDate
with engine.connect() as conn:
    try:
        conn.execute(text('CREATE INDEX idx_dt_fulldate ON Dim_Time(FullDate)'))
        conn.commit()
    except:
        conn.rollback()

# Step 1: Fact_Orders
print('\n⏳ Creating Fact_Orders...')
with engine.connect() as conn:
    conn.execute(text('DROP TABLE IF EXISTS Fact_Orders'))
    conn.commit()

with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE Fact_Orders (
            OrderID      INT AUTO_INCREMENT PRIMARY KEY,
            InvoiceNo    VARCHAR(20),
            CustomerID   INT,
            ProductID    INT,
            TimeID       INT,
            LocationID   INT,
            Quantity     INT,
            UnitPrice    DECIMAL(10,2),
            TotalPrice   DECIMAL(10,2),
            Discount     DECIMAL(5,2) DEFAULT 0,
            INDEX idx_fo_customer (CustomerID),
            INDEX idx_fo_product (ProductID),
            INDEX idx_fo_time (TimeID),
            INDEX idx_fo_location (LocationID)
        )
    """))
    conn.commit()

start = time.time()
print('  Inserting rows (using indexed JOIN)...')
with engine.connect() as conn:
    result = conn.execute(text("""
        INSERT INTO Fact_Orders (InvoiceNo, CustomerID, ProductID, TimeID, LocationID, Quantity, UnitPrice, TotalPrice, Discount)
        SELECT cr.Invoice, cr.CustomerID, dp.ProductID, dt.TimeID, dl.LocationID,
               cr.Quantity, cr.Price, cr.TotalPrice, 0
        FROM clean_retail cr
        JOIN Dim_Product dp ON cr.StockCode = dp.StockCode
        JOIN Dim_Time dt ON cr.InvoiceDay = dt.FullDate
        JOIN Dim_Location dl ON cr.Country = dl.Country
    """))
    conn.commit()
    print(f'  ✅ Fact_Orders: {result.rowcount:,} rows ({time.time()-start:.1f}s)')

# Step 2: Fact_Reviews
print('\n⏳ Creating Fact_Reviews...')
with engine.connect() as conn:
    conn.execute(text('DROP TABLE IF EXISTS Fact_Reviews'))
    conn.commit()

with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE Fact_Reviews (
            ReviewID INT AUTO_INCREMENT PRIMARY KEY,
            Rating INT, Title VARCHAR(255), ReviewText TEXT,
            ReviewLength INT, Age INT,
            DivisionName VARCHAR(50), DepartmentName VARCHAR(50),
            ClassName VARCHAR(50), CNN_Matched_Class VARCHAR(50),
            Recommended TINYINT(1), PositiveFeedback INT, SentimentScore FLOAT DEFAULT NULL
        )
    """))
    conn.commit()

start = time.time()
with engine.connect() as conn:
    result = conn.execute(text("""
        INSERT INTO Fact_Reviews (Rating, Title, ReviewText, ReviewLength, Age, DivisionName, DepartmentName, ClassName, CNN_Matched_Class, Recommended, PositiveFeedback)
        SELECT Rating, Title, `Review Text`, ReviewLength, Age,
               `Division Name`, `Department Name`, `Class Name`, CNN_Matched_Class,
               `Recommended IND`, `Positive Feedback Count`
        FROM clean_reviews
    """))
    conn.commit()
    print(f'  ✅ Fact_Reviews: {result.rowcount:,} rows ({time.time()-start:.1f}s)')

# Step 3: Final dataset
print('\n⏳ Creating final_customer_dataset...')
with engine.connect() as conn:
    conn.execute(text('DROP TABLE IF EXISTS final_customer_dataset'))
    conn.commit()

start = time.time()
with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE final_customer_dataset AS
        SELECT
            c.CustomerID, c.Country, c.JoinDate,
            c.Recency, c.Frequency, c.Monetary, c.AvgOrderValue, c.ChurnLabel,
            fo.TotalItems, fo.UniqueProducts, fo.TotalOrders,
            fo.FirstOrderDate, fo.LastOrderDate,
            DATEDIFF(fo.LastOrderDate, fo.FirstOrderDate) AS CustomerLifespanDays,
            l.Region, fo.AvgQuantityPerOrder, fo.AvgPricePerItem
        FROM Dim_Customer c
        LEFT JOIN (
            SELECT CustomerID,
                SUM(Quantity) AS TotalItems,
                COUNT(DISTINCT ProductID) AS UniqueProducts,
                COUNT(DISTINCT InvoiceNo) AS TotalOrders,
                MIN(t.FullDate) AS FirstOrderDate,
                MAX(t.FullDate) AS LastOrderDate,
                ROUND(AVG(Quantity), 2) AS AvgQuantityPerOrder,
                ROUND(AVG(UnitPrice), 2) AS AvgPricePerItem
            FROM Fact_Orders fo
            JOIN Dim_Time t ON fo.TimeID = t.TimeID
            GROUP BY CustomerID
        ) fo ON c.CustomerID = fo.CustomerID
        LEFT JOIN Dim_Location l ON c.Country = l.Country
    """))
    conn.commit()
    print(f'  ✅ Final dataset created ({time.time()-start:.1f}s)')

# Summary
print('\n' + '='*60)
print('  DATA WAREHOUSE COMPLETE!')
print('='*60)
with engine.connect() as conn:
    for t in ['Dim_Customer', 'Dim_Product', 'Dim_Time', 'Dim_Location',
              'Fact_Orders', 'Fact_Reviews', 'final_customer_dataset']:
        count = conn.execute(text(f'SELECT COUNT(*) FROM {t}')).scalar()
        print(f'  📋 {t}: {count:,} rows')

print('\n👉 Next: python scripts/03_export_final.py')
