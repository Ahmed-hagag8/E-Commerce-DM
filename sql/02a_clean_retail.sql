-- ============================================================
-- Step 2A: Clean Retail Data
-- Run this FIRST in phpMyAdmin
-- ============================================================

USE ecommerce_dm;

-- Check raw stats
SELECT 'BEFORE CLEANING' AS Step, COUNT(*) AS total_rows FROM staging_retail;

-- Create cleaned table (filter nulls, cancellations, negatives)
DROP TABLE IF EXISTS clean_retail;
CREATE TABLE clean_retail AS
SELECT * FROM staging_retail
WHERE `Customer ID` IS NOT NULL
  AND Invoice NOT LIKE 'C%'
  AND Quantity > 0
  AND Price > 0;

-- Remove duplicates using business keys (same invoice + product + customer + qty + price = duplicate)
-- DISTINCT * is unreliable because timestamp differences can make true duplicates appear unique
DROP TABLE IF EXISTS clean_retail_dedup;
CREATE TABLE clean_retail_dedup AS
SELECT Invoice, StockCode, MAX(Description) AS Description, Quantity,
       MAX(InvoiceDate) AS InvoiceDate, Price, `Customer ID`, MAX(Country) AS Country
FROM clean_retail
GROUP BY Invoice, StockCode, `Customer ID`, Quantity, Price;
DROP TABLE clean_retail;
RENAME TABLE clean_retail_dedup TO clean_retail;

SELECT 'After filter + dedup' AS Step, COUNT(*) AS total_rows FROM clean_retail;

-- Add computed columns
ALTER TABLE clean_retail
    ADD COLUMN TotalPrice DECIMAL(10,2),
    ADD COLUMN Year INT,
    ADD COLUMN Month INT,
    ADD COLUMN DayOfWeek VARCHAR(10),
    ADD COLUMN Hour INT,
    ADD COLUMN IsWeekend TINYINT(1),
    ADD COLUMN Quarter INT,
    ADD COLUMN CustomerID INT;

UPDATE clean_retail SET
    TotalPrice = Quantity * Price,
    Year = YEAR(InvoiceDate),
    Month = MONTH(InvoiceDate),
    DayOfWeek = DAYNAME(InvoiceDate),
    Hour = HOUR(InvoiceDate),
    IsWeekend = CASE WHEN DAYOFWEEK(InvoiceDate) IN (1, 7) THEN 1 ELSE 0 END,
    Quarter = QUARTER(InvoiceDate),
    CustomerID = CAST(`Customer ID` AS UNSIGNED);

-- Clean Description
UPDATE clean_retail SET Description = UPPER(TRIM(Description)) WHERE Description IS NOT NULL;
UPDATE clean_retail SET Description = 'UNKNOWN' WHERE Description IS NULL;

-- Feature Engineering: Product Category (keyword-based classification)
ALTER TABLE clean_retail ADD COLUMN ProductCategory VARCHAR(30);
UPDATE clean_retail SET ProductCategory = CASE
    WHEN Description REGEXP 'DRESS|SCARF|SHIRT|COAT|JACKET|GLOVE|SOCK|HAT|CAPE|APRON|SKIRT|VEST|SWEATER|BLOUSE'
        THEN 'Fashion & Clothing'
    WHEN Description REGEXP 'BAG|PURSE|HANDBAG|WALLET|BACKPACK|TOTE|SHOPPER|POUCH'
        THEN 'Bags & Accessories'
    WHEN Description REGEXP 'MUG|CUP|PLATE|BOWL|BOTTLE|GLASS|TRAY|TEAPOT|COFFEE|TEA SET|COASTER|NAPKIN|SPOON|FORK|JUG'
        THEN 'Kitchen & Dining'
    WHEN Description REGEXP 'CANDLE|CUSHION|FRAME|LAMP|CLOCK|MIRROR|VASE|RUG|CURTAIN|DOORMAT|HOOK|HOLDER|DRAWER|SHELF|STORAGE'
        THEN 'Home & Decor'
    WHEN Description REGEXP 'CHRISTMAS|XMAS|EASTER|BIRTHDAY|PARTY|GIFT|WEDDING|VALENTINE|HALLOWEEN|RIBBON|WRAP|BOW'
        THEN 'Gifts & Seasonal'
    WHEN Description REGEXP 'PEN|PENCIL|NOTEBOOK|JOURNAL|CARD|STICKER|STAMP|PAPER|LETTER|POSTCARD|TAG|LABEL'
        THEN 'Stationery & Cards'
    WHEN Description REGEXP 'TOY|GAME|DOLL|PUZZLE|CRAFT|PAINT|BEAD|SEWING'
        THEN 'Toys & Crafts'
    ELSE 'General'
END;

-- Category distribution
SELECT ProductCategory, COUNT(*) AS TotalRows,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM clean_retail), 1) AS Percentage
FROM clean_retail
GROUP BY ProductCategory
ORDER BY TotalRows DESC;

SELECT 'RETAIL CLEAN DONE' AS Step, COUNT(*) AS total_rows FROM clean_retail;
