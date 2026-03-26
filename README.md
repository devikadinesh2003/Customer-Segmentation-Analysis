
# 🧠 Mall Customer Segmentation Analysis




## 📌 Project Overview

This project performs customer segmentation using unsupervised machine learning techniques to identify distinct groups of customers based on their purchasing behavior.

The analysis is conducted on the Mall Customers Dataset, focusing on:

* Age
* Annual Income
* Spending Score

The goal is to help businesses understand customer patterns and improve marketing strategies.
## 🎯 Objectives

* Perform exploratory data analysis (EDA)
* Apply clustering algorithms to segment customers
* Compare clustering techniques
* Evaluate cluster quality using metrics
## ⚙️ Technologies Used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* SciPy, Statsmodels
## 📊 Exploratory Data Analysis

The dataset was analyzed using:

* Gender distribution (bar chart)
* Age distribution (histogram)
* Correlation heatmap
* Pair plots for feature relationships
## 🤖 Machine Learning Models

🔹 K-Means Clustering
* Optimal clusters selected using Elbow Method and Silhouette Score
* Customers grouped into meaningful segments
* Cluster centers analyzed in original scale

🔹 Hierarchical Clustering
* Dendrogram used for visualization
* Optimal clusters selected using silhouette score
* Compared with K-Means results
## 📈 Evaluation Metrics

* Silhouette Score – measures cluster quality
* Adjusted Rand Index (ARI) – compares clustering similarity
* Contingency Table – visual comparison of clusters
## 📊 Key Insights

* Distinct customer groups identified based on income and spending behavior
* High-income low-spending customers require targeted marketing
* Young high-spending customers form a valuable segment
* Clustering helps in personalized business strategies
## 💡 Future Improvements

* Use advanced clustering (DBSCAN, Gaussian Mixture Models)
* Deploy as a web app/dashboard
* Integrate real-time customer data
