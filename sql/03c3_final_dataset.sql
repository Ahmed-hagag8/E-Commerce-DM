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
    fo.TotalItems,
    fo.UniqueProducts,
    fo.TotalOrders,
    fo.FirstOrderDate,
    fo.LastOrderDate,
    DATEDIFF(fo.LastOrderDate, fo.FirstOrderDate) AS CustomerLifespanDays,
    l.Region,
    fo.AvgQuantityPerOrder,
    fo.AvgPricePerItem
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

SELECT 'FINAL DATASET created' AS Step, COUNT(*) AS TotalRows FROM final_customer_dataset;

-- Summary
SELECT 'Dim_Customer' AS TableName, COUNT(*) AS TotalRows FROM Dim_Customer
UNION ALL SELECT 'Dim_Product', COUNT(*) FROM Dim_Product
UNION ALL SELECT 'Dim_Time', COUNT(*) FROM Dim_Time
UNION ALL SELECT 'Dim_Location', COUNT(*) FROM Dim_Location
UNION ALL SELECT 'Fact_Orders', COUNT(*) FROM Fact_Orders
UNION ALL SELECT 'Fact_Reviews', COUNT(*) FROM Fact_Reviews
UNION ALL SELECT 'final_customer_dataset', COUNT(*) FROM final_customer_dataset;

SELECT * FROM final_customer_dataset LIMIT 10;
