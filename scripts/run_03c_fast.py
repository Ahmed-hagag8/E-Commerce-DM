from sqlalchemy import create_engine, text
import time

engine = create_engine('mysql+pymysql://root:root@localhost:3306/ecommerce_dm')

print('='*60)
print('  Step 3C — FAST version')
print('=' * 60)

print('\n⏳ Adding date index to clean_retail...')
with engine.connect() as conn:
    try:
        conn.execute(text('ALTER TABLE clean_retail ADD COLUMN InvoiceDay DATE'))
        conn.execute(text('UPDATE clean_retail SET InvoiceDay = DATE(InvoiceDate)'))
        conn.execute(text('CREATE INDEX idx_invoice_day ON clean_retail(InvoiceDay)'))
        conn.commit()
        print('  ✅ Date index added')
    except Exception as e:
        conn.rollback()
        if 'Duplicate' in str(e) or 'exists' in str(e):
            print('  ✅ Index already exists')
        else:
            print(f'  ⚠️ Index creation skipped: {str(e)[:100]}')

with engine.connect() as conn:
    try:
        conn.execute(text('CREATE INDEX idx_dt_fulldate ON Dim_Time(FullDate)'))
        conn.commit()
    except Exception as e:
        conn.rollback()

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
            INDEX idx_fo_location (LocationID),
            FOREIGN KEY (CustomerID) REFERENCES Dim_Customer(CustomerID),
            FOREIGN KEY (ProductID) REFERENCES Dim_Product(ProductID),
            FOREIGN KEY (TimeID) REFERENCES Dim_Time(TimeID),
            FOREIGN KEY (LocationID) REFERENCES Dim_Location(LocationID)
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

print('\n⏳ Creating Fact_Reviews...')
with engine.connect() as conn:
    conn.execute(text('DROP TABLE IF EXISTS Fact_Reviews'))
    conn.commit()

with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE Fact_Reviews (
            ReviewID INT AUTO_INCREMENT PRIMARY KEY,
            ClothingID INT,
            Rating INT, Title VARCHAR(255), ReviewText TEXT,
            ReviewLength INT, Age INT,
            DivisionName VARCHAR(50), DepartmentName VARCHAR(50),
            ClassName VARCHAR(50), CNN_Matched_Class VARCHAR(50),
            Recommended TINYINT(1), PositiveFeedback INT, SentimentScore FLOAT DEFAULT NULL,
            INDEX idx_fr_rating (Rating),
            INDEX idx_fr_class (CNN_Matched_Class),
            INDEX idx_fr_clothing (ClothingID)
        )
    """))
    conn.commit()

start = time.time()
with engine.connect() as conn:
    result = conn.execute(text("""
        INSERT INTO Fact_Reviews (ClothingID, Rating, Title, ReviewText, ReviewLength, Age, DivisionName, DepartmentName, ClassName, CNN_Matched_Class, Recommended, PositiveFeedback)
        SELECT `Clothing ID`, Rating, Title, `Review Text`, ReviewLength, Age,
               `Division Name`, `Department Name`, `Class Name`, CNN_Matched_Class,
               `Recommended IND`, `Positive Feedback Count`
        FROM clean_reviews
    """))
    conn.commit()
    print(f'  ✅ Fact_Reviews: {result.rowcount:,} rows ({time.time()-start:.1f}s)')

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
            COALESCE(fo.TotalItems, 0) AS TotalItems,
            COALESCE(fo.UniqueProducts, 0) AS UniqueProducts,
            COALESCE(fo.TotalOrders, 0) AS TotalOrders,
            COALESCE(fo.FirstOrderDate, c.JoinDate) AS FirstOrderDate,
            COALESCE(fo.LastOrderDate, c.JoinDate) AS LastOrderDate,
            COALESCE(DATEDIFF(fo.LastOrderDate, fo.FirstOrderDate), 0) AS CustomerLifespanDays,
            l.Region,
            COALESCE(fo.AvgQuantityPerOrder, 0) AS AvgQuantityPerOrder,
            COALESCE(fo.AvgPricePerItem, 0) AS AvgPricePerItem
        FROM Dim_Customer c
        LEFT JOIN (
            SELECT CustomerID,
                SUM(Quantity) AS TotalItems,
                COUNT(DISTINCT ProductID) AS UniqueProducts,
                COUNT(DISTINCT InvoiceNo) AS TotalOrders,
                MIN(t.FullDate) AS FirstOrderDate,
                MAX(t.FullDate) AS LastOrderDate,
                ROUND(SUM(Quantity) * 1.0 / COUNT(DISTINCT InvoiceNo), 2) AS AvgQuantityPerOrder,
                ROUND(SUM(Quantity * UnitPrice) / SUM(Quantity), 2) AS AvgPricePerItem
            FROM Fact_Orders fo
            JOIN Dim_Time t ON fo.TimeID = t.TimeID
            GROUP BY CustomerID
        ) fo ON c.CustomerID = fo.CustomerID
        LEFT JOIN Dim_Location l ON c.Country = l.Country
    """))
    conn.commit()

    nulls = conn.execute(text("""
        SELECT 
            SUM(CASE WHEN TotalItems IS NULL THEN 1 ELSE 0 END) AS null_items,
            SUM(CASE WHEN Region IS NULL THEN 1 ELSE 0 END) AS null_region
        FROM final_customer_dataset
    """)).fetchone()
    print(f'  📋 ETL Validation: NULL TotalItems={nulls[0]}, NULL Region={nulls[1]}')
    
    expected = conn.execute(text('SELECT COUNT(*) FROM Dim_Customer')).scalar()
    actual = conn.execute(text('SELECT COUNT(*) FROM final_customer_dataset')).scalar()
    status = '✅ PASS' if expected == actual else '❌ FAIL'
    print(f'  📋 Row Integrity: Expected={expected:,}, Actual={actual:,} → {status}')
    print(f'  ✅ Final dataset created ({time.time()-start:.1f}s)')

print('\n' + '='*60)
print('  DATA WAREHOUSE COMPLETE!')
print('='*60)
with engine.connect() as conn:
    for t in ['Dim_Customer', 'Dim_Product', 'Dim_Time', 'Dim_Location',
              'Fact_Orders', 'Fact_Reviews', 'final_customer_dataset']:
        count = conn.execute(text(f'SELECT COUNT(*) FROM {t}')).scalar()
        print(f'  📋 {t}: {count:,} rows')

print('\n👉 Next: python scripts/03_export_final.py')
