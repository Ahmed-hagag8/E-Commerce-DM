-- ============================================================
-- Step 3A: Create Dimension Tables
-- Run this FIRST after preprocessing
-- ============================================================

USE ecommerce_dm;
SET FOREIGN_KEY_CHECKS = 0;

-- --------------------------------------------------------
-- Dim_Time
-- --------------------------------------------------------
DROP TABLE IF EXISTS Dim_Time;
CREATE TABLE Dim_Time (
    TimeID       INT AUTO_INCREMENT PRIMARY KEY,
    FullDate     DATE NOT NULL,
    Day          INT,
    DayOfWeek    VARCHAR(10),
    Month        INT,
    MonthName    VARCHAR(15),
    Quarter      INT,
    Year         INT,
    IsWeekend    TINYINT(1) DEFAULT 0,
    IsHoliday    TINYINT(1) DEFAULT 0
);

INSERT INTO Dim_Time (FullDate, Day, DayOfWeek, Month, MonthName, Quarter, Year, IsWeekend, IsHoliday)
SELECT DISTINCT
    DATE(InvoiceDate) AS FullDate,
    DAY(InvoiceDate),
    DAYNAME(InvoiceDate),
    MONTH(InvoiceDate),
    MONTHNAME(InvoiceDate),
    QUARTER(InvoiceDate),
    YEAR(InvoiceDate),
    CASE WHEN DAYOFWEEK(InvoiceDate) IN (1, 7) THEN 1 ELSE 0 END,
    0
FROM clean_retail
ORDER BY DATE(InvoiceDate);

SELECT 'Dim_Time created' AS Step, COUNT(*) AS TotalRows FROM Dim_Time;

-- --------------------------------------------------------
-- Dim_Location
-- --------------------------------------------------------
DROP TABLE IF EXISTS Dim_Location;
CREATE TABLE Dim_Location (
    LocationID   INT AUTO_INCREMENT PRIMARY KEY,
    Country      VARCHAR(50) NOT NULL,
    Region       VARCHAR(50),
    City         VARCHAR(50) DEFAULT 'N/A'
);

INSERT INTO Dim_Location (Country, Region)
SELECT DISTINCT Country,
    CASE
        WHEN Country IN ('United Kingdom','France','Germany','Spain','Italy','Netherlands',
            'Belgium','Switzerland','Portugal','Norway','Sweden','Denmark','Finland',
            'Austria','Ireland','Poland','Czech Republic','Greece','Iceland','Malta',
            'Cyprus','Lithuania','Channel Islands','EIRE','European Community') THEN 'Europe'
        WHEN Country IN ('USA','Canada') THEN 'North America'
        WHEN Country = 'Brazil' THEN 'South America'
        WHEN Country = 'Australia' THEN 'Oceania'
        WHEN Country IN ('Japan','Singapore','Hong Kong') THEN 'Asia'
        WHEN Country IN ('Israel','Lebanon','United Arab Emirates','Bahrain','Saudi Arabia') THEN 'Middle East'
        WHEN Country IN ('South Africa','Nigeria','RSA') THEN 'Africa'
        ELSE 'Other'
    END AS Region
FROM clean_retail;

SELECT 'Dim_Location created' AS Step, COUNT(*) AS TotalRows FROM Dim_Location;

-- --------------------------------------------------------
-- Dim_Product
-- --------------------------------------------------------
DROP TABLE IF EXISTS Dim_Product;
CREATE TABLE Dim_Product (
    ProductID     INT AUTO_INCREMENT PRIMARY KEY,
    StockCode     VARCHAR(20),
    ProductName   VARCHAR(255),
    Category      VARCHAR(50) DEFAULT 'General',
    SubCategory   VARCHAR(50) DEFAULT 'N/A',
    AvgPrice      DECIMAL(10,2),
    AvgRating     FLOAT DEFAULT 0,
    ImageCategory VARCHAR(50) DEFAULT NULL,
    PriceRange    VARCHAR(20)
);

INSERT INTO Dim_Product (StockCode, ProductName, Category, AvgPrice, PriceRange)
SELECT 
    StockCode,
    MAX(Description) AS ProductName,
    MAX(ProductCategory) AS Category,  -- MAX is deterministic (vs ANY_VALUE which is random)
    ROUND(AVG(Price), 2) AS AvgPrice,
    CASE 
        WHEN AVG(Price) <= 2 THEN 'Low'
        WHEN AVG(Price) <= 5 THEN 'Medium'
        WHEN AVG(Price) <= 15 THEN 'High'
        ELSE 'Premium'
    END AS PriceRange
FROM clean_retail
GROUP BY StockCode;

SELECT 'Dim_Product created' AS Step, COUNT(*) AS TotalRows FROM Dim_Product;

SELECT 'STEP 3A DONE - All dimensions created' AS Result;
