# 📊 Software Mailing Analytics – North-Point Direct Mail Optimization

**Author:** Reishekesh Reddy Inavola
**Course:** CSDA 6010 – Analytics Practicum
---

## 📌 Project Overview

This project focuses on improving the effectiveness of **North-Point’s direct mail marketing campaign** using **data-driven analytics**.
The company faced a **low response rate (5.3%)** and high marketing costs due to poor targeting.
Using **classification and clustering techniques**, this project identifies high-probability buyers and meaningful customer segments to maximize **ROI and sales performance**.

---

## 🎯 Business Objectives

* Increase customer response rate from **5.3% to at least 8%**
* Improve **marketing ROI by 20%**
* Increase **overall sales by 15%**
* Reduce wasted mailings to uninterested customers
* Segment customers for targeted marketing strategies

---

## 🧠 Analytical Goals

1. **Classification**

   * Predict the likelihood of customer purchase (`Purchase`)
   * Identify key drivers of response behavior

2. **Customer Segmentation**

   * Group customers based on spending, frequency, and recency
   * Enable targeted and personalized marketing campaigns

---

## 🗂 Dataset Description

* **Records:** 2,000 customers (balanced sample)
* **Target Variable:** `Purchase` (Yes / No)
* **Key Predictors:**

  * Transaction frequency (`Freq`)
  * Spending
  * Source channels
  * Web order behavior
  * Recency variables
* **Source Channels:** 15 binary indicators consolidated into one `source` variable
* **Missing Values:** None

---

## 🔍 Methodology

### 🔹 Data Preparation

* Zero-value and consistency checks
* Factor conversion for categorical variables
* Source variables consolidated into a single column
* Feature normalization for clustering

---

### 🔹 Exploratory Data Analysis (EDA)

* Spending and frequency distributions
* Purchase vs non-purchase behavior
* Correlation analysis
* Chi-square tests for categorical predictors
* T-tests for numeric predictors

---

## 🤖 Models Implemented

### 1️⃣ Classification Models

#### 🌳 Decision Tree (CART)

* Accuracy: **83.5%**
* Sensitivity: **87%**
* Specificity: **80%**
* Highly interpretable rules for business decisions

#### 🌲 Random Forest

* Accuracy: **83.25%**
* Sensitivity: **91.5%**
* Precision: **78.5%**
* Best model for identifying true buyers
* Feature importance highlights key predictors:

  * Frequency
  * Source
  * Recency variables

> **Note:** `Spending` was removed during classification to avoid data leakage.

---

### 2️⃣ Clustering Models

#### 🔗 Hierarchical Clustering (Gower Distance)

* Optimal clusters: **8**
* Mixed data handling (numeric + categorical)
* Useful for exploratory segmentation

#### 🔵 K-Means Clustering

* Optimal clusters: **3**
* Silhouette score: **0.589**
* Clear customer segments:

  * High-value loyal customers
  * Moderate-value customers
  * Inactive/dormant customers

---

## 📈 Key Business Insights

* **Frequency and recency** are the strongest predictors of purchase
* Web-order customers are significantly more likely to buy
* A small group of high-value customers contributes disproportionately to revenue
* Targeted campaigns can significantly reduce mailing costs
* Adjusted probabilities reflect real-world response rates (3%–10%)

---

## 💡 Business Value

* Enables **targeted mailing** instead of mass campaigns
* Reduces marketing waste
* Improves customer engagement and retention
* Supports long-term competitive advantage through data-driven decisions

---

## 🛠 Technologies & Libraries

* **Language:** R
* **Libraries Used:**

  * `dplyr`, `ggplot2`, `corrplot`
  * `rpart`, `rpart.plot`
  * `randomForest`, `caret`
  * `cluster`, `caTools`

---

## 📁 Repository Structure

```
├── North-point r code.R
├── Software_Mailing_Analytics.docx
├── Software_Mailing_Analytics_Presentation.pptx
└── README.md
```

---

## 📌 How to Run

1. Open R or RStudio
2. Install required packages
3. Update the dataset file path
4. Run `North-point r code.R` sequentially

---

## 📊 Expected Outcomes

* Increase response rate to **≥ 8%**
* Improve ROI by **~20%**
* Boost sales by **~15%**
* Enable smarter, behavior-driven marketing campaigns

---

## ✅ Conclusion

This project demonstrates how **classification and clustering analytics** can transform a costly, ineffective direct mail strategy into a **high-ROI, targeted marketing system**.
By focusing on customer behavior rather than demographics, North-Point can achieve sustainable growth and competitive advantage.
