-- ============================================================
-- Step 3B: RFM + Dim_Customer (OPTIMIZED)
-- Run this SECOND
-- ============================================================

USE ecommerce_dm;

-- Reference date = 1 day after last transaction
SET @ref_date = (SELECT DATE_ADD(MAX(DATE(InvoiceDate)), INTERVAL 1 DAY) FROM clean_retail);
SELECT @ref_date AS ReferenceDate;

-- Create RFM table with country + join date in ONE pass (fast!)
DROP TABLE IF EXISTS rfm_table;
CREATE TABLE rfm_table AS
SELECT
    CustomerID,
    DATEDIFF(@ref_date, MAX(DATE(InvoiceDate))) AS Recency,
    COUNT(DISTINCT Invoice) AS Frequency,
    ROUND(SUM(TotalPrice), 2) AS Monetary,
    MIN(DATE(InvoiceDate)) AS JoinDate
FROM clean_retail
GROUP BY CustomerID;

-- Add computed columns
ALTER TABLE rfm_table
    ADD COLUMN AvgOrderValue DECIMAL(10,2),
    ADD COLUMN ChurnLabel TINYINT(1);

UPDATE rfm_table SET
    AvgOrderValue = ROUND(Monetary / Frequency, 2),
    ChurnLabel = CASE WHEN Recency > 90 THEN 1 ELSE 0 END;

SELECT 'RFM computed' AS Step, COUNT(*) AS Customers FROM rfm_table;
SELECT 
    SUM(ChurnLabel) AS Churned,
    SUM(1 - ChurnLabel) AS Active,
    ROUND(SUM(ChurnLabel) * 100.0 / COUNT(*), 1) AS ChurnRatePercent
FROM rfm_table;

-- Pre-compute most common country per customer (fast GROUP BY)
DROP TABLE IF EXISTS customer_country;
CREATE TABLE customer_country AS
SELECT CustomerID, Country, COUNT(*) AS cnt
FROM clean_retail
GROUP BY CustomerID, Country;

-- Keep only the top country per customer
DROP TABLE IF EXISTS customer_top_country;
CREATE TABLE customer_top_country AS
SELECT cc.CustomerID, ANY_VALUE(cc.Country) AS Country
FROM customer_country cc
INNER JOIN (
    SELECT CustomerID, MAX(cnt) AS max_cnt
    FROM customer_country
    GROUP BY CustomerID
) mx ON cc.CustomerID = mx.CustomerID AND cc.cnt = mx.max_cnt
GROUP BY cc.CustomerID;

-- --------------------------------------------------------
-- Dim_Customer (fast — no subqueries!)
-- --------------------------------------------------------
DROP TABLE IF EXISTS Dim_Customer;
CREATE TABLE Dim_Customer (
    CustomerID    INT PRIMARY KEY,
    Country       VARCHAR(50),
    AgeGroup      VARCHAR(20) DEFAULT 'Unknown',
    JoinDate      DATE,
    Recency       INT,
    Frequency     INT,
    Monetary      DECIMAL(10,2),
    AvgOrderValue DECIMAL(10,2),
    RFM_Segment   VARCHAR(30),
    ChurnLabel    TINYINT(1)
);

INSERT INTO Dim_Customer (CustomerID, Country, JoinDate, Recency, Frequency, Monetary, AvgOrderValue, ChurnLabel)
SELECT 
    r.CustomerID,
    c.Country,
    r.JoinDate,
    r.Recency,
    r.Frequency,
    r.Monetary,
    r.AvgOrderValue,
    r.ChurnLabel
FROM rfm_table r
LEFT JOIN customer_top_country c ON r.CustomerID = c.CustomerID;

-- Cleanup temp tables
DROP TABLE IF EXISTS customer_country;
DROP TABLE IF EXISTS customer_top_country;

SELECT 'Dim_Customer created' AS Step, COUNT(*) AS TotalRows FROM Dim_Customer;
SELECT 'STEP 3B DONE' AS Result;
