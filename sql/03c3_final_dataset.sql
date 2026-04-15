-- ============================================================
-- Step 3C-3: Create Final Integrated Dataset
-- Run this LAST in phpMyAdmin
-- ============================================================

USE ecommerce_dm;

DROP TABLE IF EXISTS final_customer_dataset;
CREATE TABLE final_customer_dataset AS
SELECT
    c.CustomerID,
    c.Country,
    c.JoinDate,
    c.Recency,
    c.Frequency,
    c.Monetary,
    c.AvgOrderValue,
    c.ChurnLabel,
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
    SELECT 
        CustomerID,
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
LEFT JOIN Dim_Location l ON c.Country = l.Country;

-- ============================================================
-- ETL Validation: Full Pipeline Integrity Check
-- ============================================================
SELECT '=== FULL ETL VALIDATION ===' AS Step;

-- Check for NULL values in critical columns
SELECT 'NULL CHECK' AS Step,
    SUM(CASE WHEN TotalItems IS NULL THEN 1 ELSE 0 END) AS Null_TotalItems,
    SUM(CASE WHEN UniqueProducts IS NULL THEN 1 ELSE 0 END) AS Null_UniqueProducts,
    SUM(CASE WHEN Region IS NULL THEN 1 ELSE 0 END) AS Null_Region,
    SUM(CASE WHEN ChurnLabel IS NULL THEN 1 ELSE 0 END) AS Null_ChurnLabel
FROM final_customer_dataset;

-- Row count consistency across all tables
SELECT 'Dim_Customer' AS TableName, COUNT(*) AS TotalRows FROM Dim_Customer
UNION ALL SELECT 'Dim_Product', COUNT(*) FROM Dim_Product
UNION ALL SELECT 'Dim_Time', COUNT(*) FROM Dim_Time
UNION ALL SELECT 'Dim_Location', COUNT(*) FROM Dim_Location
UNION ALL SELECT 'Fact_Orders', COUNT(*) FROM Fact_Orders
UNION ALL SELECT 'Fact_Reviews', COUNT(*) FROM Fact_Reviews
UNION ALL SELECT 'final_customer_dataset', COUNT(*) FROM final_customer_dataset;

-- Verify final dataset matches Dim_Customer count (no data loss)
SELECT 
    (SELECT COUNT(*) FROM Dim_Customer) AS Expected_Customers,
    (SELECT COUNT(*) FROM final_customer_dataset) AS Actual_Customers,
    CASE 
        WHEN (SELECT COUNT(*) FROM Dim_Customer) = (SELECT COUNT(*) FROM final_customer_dataset)
        THEN 'PASS - No data loss'
        ELSE 'FAIL - Data loss detected!'
    END AS Validation_Result;

SELECT * FROM final_customer_dataset LIMIT 10;
