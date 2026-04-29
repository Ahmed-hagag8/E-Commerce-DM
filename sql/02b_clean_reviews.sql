-- ============================================================
-- Step 2B: Clean Reviews
-- Run this SECOND in phpMyAdmin
-- ============================================================

USE ecommerce_dm;

-- ============================================================
-- CLEAN REVIEWS
-- ============================================================
DROP TABLE IF EXISTS clean_reviews;
CREATE TABLE clean_reviews AS
SELECT * FROM staging_reviews;

-- Drop unnamed index column
-- Drop unnamed index column (ignore error if column doesn't exist)
SET @col_exists = (SELECT COUNT(*) FROM information_schema.columns WHERE table_schema='ecommerce_dm' AND table_name='clean_reviews' AND column_name='Unnamed: 0');
SET @sql = IF(@col_exists > 0, 'ALTER TABLE clean_reviews DROP COLUMN `Unnamed: 0`', 'SELECT 1');
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Delete rows with no review text
DELETE FROM clean_reviews 
WHERE `Review Text` IS NULL OR TRIM(`Review Text`) = '';

-- Clean text
UPDATE clean_reviews SET `Review Text` = LOWER(TRIM(`Review Text`));

-- Fill missing values
UPDATE clean_reviews SET Title = 'Unknown' WHERE Title IS NULL OR TRIM(Title) = '';
UPDATE clean_reviews SET `Division Name` = 'Unknown' WHERE `Division Name` IS NULL;
UPDATE clean_reviews SET `Department Name` = 'Unknown' WHERE `Department Name` IS NULL;
UPDATE clean_reviews SET `Class Name` = 'Unknown' WHERE `Class Name` IS NULL;

-- Add ReviewLength and CNN Category Mapping
ALTER TABLE clean_reviews 
    ADD COLUMN ReviewLength INT,
    ADD COLUMN CNN_Matched_Class VARCHAR(50);

UPDATE clean_reviews SET
    ReviewLength = (LENGTH(`Review Text`) - LENGTH(REPLACE(`Review Text`, ' ', '')) + 1),
    CNN_Matched_Class = CASE
        WHEN `Class Name` = 'Dresses' THEN 'Gaun'
        WHEN `Class Name` IN ('Knits', 'Fine gauge') THEN 'Kaos'
        WHEN `Class Name` = 'Blouses' THEN 'Kemeja'
        WHEN `Class Name` = 'Sweaters' THEN 'Sweter'
        WHEN `Class Name` = 'Pants' THEN 'Celana_Panjang'
        WHEN `Class Name` = 'Jeans' THEN 'Jeans'
        WHEN `Class Name` = 'Skirts' THEN 'Rok'
        WHEN `Class Name` = 'Jackets' THEN 'Jaket'
        WHEN `Class Name` = 'Outerwear' THEN 'Mantel'
        WHEN `Class Name` = 'Shorts' THEN 'Celana_Pendek'
        ELSE 'Lainnya' -- Means Others
    END;

SELECT 'REVIEWS CLEAN DONE' AS Step, COUNT(*) AS total_rows FROM clean_reviews;



-- ============================================================
-- SUMMARY
-- ============================================================
SELECT 'CLEANING SUMMARY' AS Info;
SELECT 'clean_retail' AS TableName, COUNT(*) AS TotalRows FROM clean_retail
UNION ALL
SELECT 'clean_reviews', COUNT(*) FROM clean_reviews;
