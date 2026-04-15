-- ============================================================
-- Step 3C-2: Create Fact_Reviews (Connected to Star Schema)
-- Run this in phpMyAdmin
-- ============================================================

USE ecommerce_dm;

DROP TABLE IF EXISTS Fact_Reviews;
CREATE TABLE Fact_Reviews (
    ReviewID       INT AUTO_INCREMENT PRIMARY KEY,
    ClothingID     INT,
    Rating         INT,
    Title          VARCHAR(255),
    ReviewText     TEXT,
    ReviewLength   INT,
    Age            INT,
    DivisionName   VARCHAR(50),
    DepartmentName VARCHAR(50),
    ClassName      VARCHAR(50),
    CNN_Matched_Class VARCHAR(50),
    Recommended    TINYINT(1),
    PositiveFeedback INT,
    SentimentScore FLOAT DEFAULT NULL,
    INDEX idx_fr_rating (Rating),
    INDEX idx_fr_class (CNN_Matched_Class),
    INDEX idx_fr_clothing (ClothingID)
);

INSERT INTO Fact_Reviews (ClothingID, Rating, Title, ReviewText, ReviewLength, Age, DivisionName, DepartmentName, ClassName, CNN_Matched_Class, Recommended, PositiveFeedback)
SELECT
    `Clothing ID`,
    Rating,
    Title,
    `Review Text`,
    ReviewLength,
    Age,
    `Division Name`,
    `Department Name`,
    `Class Name`,
    CNN_Matched_Class,
    `Recommended IND`,
    `Positive Feedback Count`
FROM clean_reviews;

-- ETL Validation: verify row counts between stages
SELECT 'ETL VALIDATION' AS Step;
SELECT
    (SELECT COUNT(*) FROM clean_reviews) AS Source_Rows,
    (SELECT COUNT(*) FROM Fact_Reviews) AS Loaded_Rows,
    (SELECT COUNT(*) FROM clean_reviews) - (SELECT COUNT(*) FROM Fact_Reviews) AS Rows_Lost,
    ROUND((SELECT COUNT(*) FROM Fact_Reviews) * 100.0 / (SELECT COUNT(*) FROM clean_reviews), 2) AS Load_Pct;

SELECT 'Fact_Reviews created' AS Step, COUNT(*) AS TotalRows FROM Fact_Reviews;
