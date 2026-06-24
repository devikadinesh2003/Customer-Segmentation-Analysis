
#  Mall Customer Segmentation Analysis using Clustering Techniques




##  Project Overview

Customer segmentation is a critical business strategy that helps organizations understand customer behavior and deliver personalized experiences. This project applies K-Means Clustering and Hierarchical Clustering to identify distinct customer groups based on demographic and spending characteristics.

The analysis focuses on discovering hidden patterns among customers and transforming raw customer data into actionable business insights.

## Business Problem

Businesses often treat all customers similarly, leading to inefficient marketing campaigns and lower customer engagement.

This project aims to:

* Identify groups of customers with similar purchasing behavior
* Discover high-value customer segments
* Support targeted marketing and customer retention strategies
* Improve business decision-making through data-driven segmentation

## Dataset 

The analysis uses the Mall Customers Dataset containing:

| Feature                | Description                       |
| ---------------------- | --------------------------------- |
| CustomerID             | Unique customer identifier        |
| Gender                 | Male/Female                       |
| Age                    | Customer age                      |
| Annual Income (k$)     | Annual income in thousand dollars |
| Spending Score (1-100) | Customer spending behavior score  |


# Dataset Statistics

| Metric              | Value                            |
| ------------------- | -------------------------------- |
| Total Customers     | 200                              |
| Total Features      | 5                                |
| Missing Values      | 0                                |
| Clustering Features | 3                                |
| Algorithms Used     | K-Means, Hierarchical Clustering |

## Exploratory Data Analysis

Performed detailed EDA using:

   * Gender distribution analysis
   * Age distribution histogram
   * Correlation heatmap
   * Pair plots
   * Income vs Spending pattern visualization
Key Findings
   * Customers belong to diverse age groups.
   * Spending behavior varies significantly across income levels.
   * Annual Income and Spending Score show strong segmentation potential.
   * Several visually distinguishable customer groups are present before clustering.
 ## Data Preprocessing

The following preprocessing steps were applied:

* Missing value handling

* Label Encoding
   * Female = 0
   * Male = 1
* Feature Scaling using StandardScaler

* Selection of clustering variables
  * Age
  * Annual Income (k$)
  * Spending Score (1-100)
  
##  Machine Learning Models

*  K-Means Clustering
   * Optimal clusters selected using Elbow Method and Silhouette Score
   * Customers grouped into meaningful segments
   * Cluster centers analyzed in original scale
 
 Results
 | Metric           | Value                 |
| ---------------- | --------------------- |
| Optimal Clusters | 6 |
| Silhouette Score | 0.4284 |

* Hierarchical Clustering
   * Dendrogram used for visualization
   * Optimal clusters selected using silhouette score
   * Compared with K-Means results

 Results
| Metric           | Value                 |
| ---------------- | --------------------- |
| Optimal Clusters | 6 |
| Silhouette Score | 0.4201 |

## Model Comparison

| Metric           | K-Means | Hierarchical |
| ---------------- | ------- | ------------ |
| Optimal Clusters | 6     | 6          |
| Silhouette Score | 0.4284     | 0.4201           |

* Adjusted Rand Index (ARI): 0.87
  * Indicates strong agreement between K-Means and Hierarchical cluster assignments.

## Customer Segments

The clustering process identifies customer groups such as:

* Premium Customers
   * High Income
   * High Spending
* Careful Customers
   * High Income
   * Low Spending
* Standard Customers
   * Moderate Income
   * Moderate Spending
* Young Big Spenders
   * Lower Income
   * High Spending
* Conservative Customers
   * Lower Income
   * Lower Spending

## Business Insights

The analysis enables businesses to:

* Identify premium customers for loyalty programs
* Target marketing campaigns more effectively
* Reduce customer acquisition costs
* Improve customer retention strategies
* Personalize offers based on customer segments

## Technologies Used
* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* SciPy

## Project Outcomes

* Segmented 200 customers into meaningful behavioral groups

* Compared K-Means and Hierarchical Clustering approaches

* Evaluated clustering quality using Silhouette Score

* Generated actionable customer insights for marketing optimization

