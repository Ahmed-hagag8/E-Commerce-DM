-- ============================================================
-- E-Commerce Data Mining — Analytical Queries
-- Database: MySQL
-- ============================================================

USE ecommerce_dm;

-- ============================================================
-- 1. Top 10 Best-Selling Products
-- ============================================================
SELECT 
    p.ProductName, 
    p.Category,
    SUM(f.Quantity) AS TotalUnitsSold,
    SUM(f.TotalPrice) AS TotalRevenue
FROM Fact_Orders f
JOIN Dim_Product p ON f.ProductID = p.ProductID
GROUP BY p.ProductID, p.ProductName, p.Category
ORDER BY TotalUnitsSold DESC
LIMIT 10;

-- ============================================================
-- 2. Monthly Revenue Trend
-- ============================================================
SELECT 
    t.Year, 
    t.Month, 
    t.MonthName,
    SUM(f.TotalPrice) AS Revenue,
    COUNT(DISTINCT f.InvoiceNo) AS TotalOrders,
    COUNT(DISTINCT f.CustomerID) AS UniqueCustomers
FROM Fact_Orders f
JOIN Dim_Time t ON f.TimeID = t.TimeID
GROUP BY t.Year, t.Month, t.MonthName
ORDER BY t.Year, t.Month;

-- ============================================================
-- 3. Customer Lifetime Value (CLV) — Top 20
-- ============================================================
SELECT 
    c.CustomerID, 
    c.RFM_Segment,
    c.Country,
    SUM(f.TotalPrice) AS LifetimeValue,
    COUNT(DISTINCT f.InvoiceNo) AS TotalOrders,
    ROUND(AVG(f.TotalPrice), 2) AS AvgOrderValue,
    c.Recency,
    c.Frequency,
    c.Monetary
FROM Fact_Orders f
JOIN Dim_Customer c ON f.CustomerID = c.CustomerID
GROUP BY c.CustomerID, c.RFM_Segment, c.Country, c.Recency, c.Frequency, c.Monetary
ORDER BY LifetimeValue DESC
LIMIT 20;

-- ============================================================
-- 4. Churn Rate by RFM Segment
-- ============================================================
SELECT 
    RFM_Segment,
    COUNT(*) AS TotalCustomers,
    SUM(ChurnLabel) AS ChurnedCustomers,
    ROUND(SUM(ChurnLabel) * 100.0 / COUNT(*), 2) AS ChurnRatePercent
FROM Dim_Customer
WHERE RFM_Segment IS NOT NULL
GROUP BY RFM_Segment
ORDER BY ChurnRatePercent DESC;

-- ============================================================
-- 5. Product Co-occurrence Pairs (for Association Rules Validation)
-- ============================================================
SELECT 
    a.ProductID AS Product_A, 
    pa.ProductName AS ProductName_A,
    b.ProductID AS Product_B, 
    pb.ProductName AS ProductName_B,
    COUNT(*) AS CoOccurrence
FROM Fact_Orders a
JOIN Fact_Orders b ON a.InvoiceNo = b.InvoiceNo AND a.ProductID < b.ProductID
JOIN Dim_Product pa ON a.ProductID = pa.ProductID
JOIN Dim_Product pb ON b.ProductID = pb.ProductID
GROUP BY a.ProductID, pa.ProductName, b.ProductID, pb.ProductName
HAVING COUNT(*) > 5
ORDER BY CoOccurrence DESC
LIMIT 20;

-- ============================================================
-- 6. Average Sentiment by Product Category
-- ============================================================
SELECT 
    p.Category,
    ROUND(AVG(r.SentimentScore), 3) AS AvgSentiment,
    ROUND(AVG(r.Rating), 2) AS AvgRating,
    COUNT(*) AS ReviewCount,
    SUM(r.IsRecommended) AS RecommendedCount,
    ROUND(SUM(r.IsRecommended) * 100.0 / COUNT(*), 1) AS RecommendRatePercent
FROM Fact_Reviews r
JOIN Dim_Product p ON r.ProductID = p.ProductID
GROUP BY p.Category
ORDER BY AvgSentiment DESC;

-- ============================================================
-- 7. Revenue by Country (Top 10)
-- ============================================================
SELECT 
    l.Country,
    SUM(f.TotalPrice) AS TotalRevenue,
    COUNT(DISTINCT f.CustomerID) AS UniqueCustomers,
    COUNT(DISTINCT f.InvoiceNo) AS TotalOrders
FROM Fact_Orders f
JOIN Dim_Location l ON f.LocationID = l.LocationID
GROUP BY l.Country
ORDER BY TotalRevenue DESC
LIMIT 10;

-- ============================================================
-- 8. Sales by Day of Week
-- ============================================================
SELECT 
    t.DayOfWeek,
    COUNT(DISTINCT f.InvoiceNo) AS TotalOrders,
    SUM(f.TotalPrice) AS TotalRevenue,
    ROUND(AVG(f.TotalPrice), 2) AS AvgOrderValue
FROM Fact_Orders f
JOIN Dim_Time t ON f.TimeID = t.TimeID
GROUP BY t.DayOfWeek
ORDER BY FIELD(t.DayOfWeek, 'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday');

-- ============================================================
-- 9. Customer Segment Profile Summary
-- ============================================================
SELECT 
    RFM_Segment,
    COUNT(*) AS CustomerCount,
    ROUND(AVG(Recency), 1) AS AvgRecency,
    ROUND(AVG(Frequency), 1) AS AvgFrequency,
    ROUND(AVG(Monetary), 2) AS AvgMonetary,
    ROUND(MIN(Monetary), 2) AS MinMonetary,
    ROUND(MAX(Monetary), 2) AS MaxMonetary
FROM Dim_Customer
WHERE RFM_Segment IS NOT NULL
GROUP BY RFM_Segment
ORDER BY AvgMonetary DESC;

-- ============================================================
-- 10. Product Price Range Distribution
-- ============================================================
SELECT 
    PriceRange,
    COUNT(*) AS ProductCount,
    ROUND(AVG(AvgRating), 2) AS AvgRating
FROM Dim_Product
WHERE PriceRange IS NOT NULL
GROUP BY PriceRange
ORDER BY FIELD(PriceRange, 'Low', 'Medium', 'High', 'Premium');
