-- ============================================================
-- Step 3C-2: Create Fact_Reviews
-- Run this in phpMyAdmin
-- ============================================================

USE ecommerce_dm;

DROP TABLE IF EXISTS Fact_Reviews;
CREATE TABLE Fact_Reviews (
    ReviewID       INT AUTO_INCREMENT PRIMARY KEY,
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
    SentimentScore FLOAT DEFAULT NULL
);

INSERT INTO Fact_Reviews (Rating, Title, ReviewText, ReviewLength, Age, DivisionName, DepartmentName, ClassName, CNN_Matched_Class, Recommended, PositiveFeedback)
SELECT
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

SELECT 'Fact_Reviews created' AS Step, COUNT(*) AS TotalRows FROM Fact_Reviews;
