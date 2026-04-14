-- ============================================================
-- Step 3C-1: Create Fact_Orders
-- Run this in phpMyAdmin
-- ============================================================

USE ecommerce_dm;

DROP TABLE IF EXISTS Fact_Orders;
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
);

INSERT INTO Fact_Orders (InvoiceNo, CustomerID, ProductID, TimeID, LocationID, Quantity, UnitPrice, TotalPrice, Discount)
SELECT
    cr.Invoice,
    cr.CustomerID,
    dp.ProductID,
    dt.TimeID,
    dl.LocationID,
    cr.Quantity,
    cr.Price,
    cr.TotalPrice,
    0
FROM clean_retail cr
JOIN Dim_Product dp ON cr.StockCode = dp.StockCode
JOIN Dim_Time dt ON DATE(cr.InvoiceDate) = dt.FullDate
JOIN Dim_Location dl ON cr.Country = dl.Country;

SELECT 'Fact_Orders created' AS Step, COUNT(*) AS TotalRows FROM Fact_Orders;
