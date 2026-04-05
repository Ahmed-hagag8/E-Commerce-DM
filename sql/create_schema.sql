-- ============================================================
-- E-Commerce Data Mining — Star Schema
-- Database: MySQL (phpMyAdmin)
-- Course: AIE 323 — Data Mining
-- ============================================================

-- Create Database
CREATE DATABASE IF NOT EXISTS ecommerce_dm;
USE ecommerce_dm;

-- ============================================================
-- DIMENSION TABLES
-- ============================================================

-- Dim_Customer
CREATE TABLE IF NOT EXISTS Dim_Customer (
    CustomerID      INT PRIMARY KEY,
    AgeGroup        VARCHAR(20),
    Country         VARCHAR(50),
    JoinDate        DATE,
    Recency         INT           COMMENT 'Days since last purchase',
    Frequency       INT           COMMENT 'Total number of orders',
    Monetary        DECIMAL(10,2) COMMENT 'Total amount spent',
    RFM_Segment     VARCHAR(30)   COMMENT 'K-Means cluster label',
    ChurnLabel      TINYINT(1)    COMMENT '1 = churned, 0 = active'
);

-- Dim_Product
CREATE TABLE IF NOT EXISTS Dim_Product (
    ProductID       INT AUTO_INCREMENT PRIMARY KEY,
    StockCode       VARCHAR(20),
    ProductName     VARCHAR(255),
    Category        VARCHAR(50),
    SubCategory     VARCHAR(50),
    AvgRating       FLOAT         DEFAULT 0,
    ImageCategory   VARCHAR(50)   COMMENT 'CNN-predicted category',
    PriceRange      VARCHAR(20)   COMMENT 'Low / Medium / High / Premium'
);

-- Dim_Time
CREATE TABLE IF NOT EXISTS Dim_Time (
    TimeID          INT AUTO_INCREMENT PRIMARY KEY,
    FullDate        DATE          NOT NULL,
    Day             INT,
    DayOfWeek       VARCHAR(10),
    Month           INT,
    MonthName       VARCHAR(15),
    Quarter         INT,
    Year            INT,
    IsWeekend       TINYINT(1)    DEFAULT 0,
    IsHoliday       TINYINT(1)    DEFAULT 0
);

-- Dim_Location
CREATE TABLE IF NOT EXISTS Dim_Location (
    LocationID      INT AUTO_INCREMENT PRIMARY KEY,
    Country         VARCHAR(50)   NOT NULL,
    Region          VARCHAR(50),
    City            VARCHAR(50)
);

-- ============================================================
-- FACT TABLES
-- ============================================================

-- Fact_Orders
CREATE TABLE IF NOT EXISTS Fact_Orders (
    OrderID         INT AUTO_INCREMENT PRIMARY KEY,
    InvoiceNo       VARCHAR(20),
    CustomerID      INT,
    ProductID       INT,
    TimeID          INT,
    LocationID      INT,
    Quantity         INT,
    UnitPrice       DECIMAL(10,2),
    TotalPrice      DECIMAL(10,2),
    Discount        DECIMAL(5,2)  DEFAULT 0,

    FOREIGN KEY (CustomerID) REFERENCES Dim_Customer(CustomerID),
    FOREIGN KEY (ProductID)  REFERENCES Dim_Product(ProductID),
    FOREIGN KEY (TimeID)     REFERENCES Dim_Time(TimeID),
    FOREIGN KEY (LocationID) REFERENCES Dim_Location(LocationID)
);

-- Fact_Reviews
CREATE TABLE IF NOT EXISTS Fact_Reviews (
    ReviewID        INT AUTO_INCREMENT PRIMARY KEY,
    CustomerID      INT,
    ProductID       INT,
    TimeID          INT,
    Rating          INT           COMMENT '1-5 star rating',
    SentimentScore  FLOAT         COMMENT 'NLP sentiment: -1 to +1',
    ReviewLength    INT           COMMENT 'Word count',
    IsRecommended   TINYINT(1)    COMMENT '1 = recommended',

    FOREIGN KEY (CustomerID) REFERENCES Dim_Customer(CustomerID),
    FOREIGN KEY (ProductID)  REFERENCES Dim_Product(ProductID),
    FOREIGN KEY (TimeID)     REFERENCES Dim_Time(TimeID)
);

-- ============================================================
-- INDEXES (for query performance)
-- ============================================================

CREATE INDEX idx_fact_orders_customer ON Fact_Orders(CustomerID);
CREATE INDEX idx_fact_orders_product  ON Fact_Orders(ProductID);
CREATE INDEX idx_fact_orders_time     ON Fact_Orders(TimeID);
CREATE INDEX idx_fact_reviews_customer ON Fact_Reviews(CustomerID);
CREATE INDEX idx_fact_reviews_product  ON Fact_Reviews(ProductID);
CREATE INDEX idx_dim_customer_segment ON Dim_Customer(RFM_Segment);
CREATE INDEX idx_dim_customer_churn   ON Dim_Customer(ChurnLabel);
CREATE INDEX idx_dim_product_category ON Dim_Product(Category);
CREATE INDEX idx_dim_time_date        ON Dim_Time(FullDate);
CREATE INDEX idx_dim_time_year_month  ON Dim_Time(Year, Month);
