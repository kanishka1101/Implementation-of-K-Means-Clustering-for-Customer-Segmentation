# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
   
2. Read the dataset using pd.read_csv().

3. Select the columns Annual Income and Spending Score.

4. Create the K-Means model with 5 clusters.

5. Apply fit_predict() to form clusters.

6. Store cluster values in the dataset.

7. Plot the clusters and centroids using matplotlib.

8. Display the graph using plt.show().
 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Kanishka G
RegisterNumber:212225040168  
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv("Mall_Customers.csv")

X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

kmeans = KMeans(n_clusters=5, random_state=1)

data['Cluster'] = kmeans.fit_predict(X)
print(data.head())
plt.scatter(X[data['Cluster']==0]['Annual Income (k$)'],
            X[data['Cluster']==0]['Spending Score (1-100)'],
            color='blue', label='Cluster 0')

plt.scatter(X[data['Cluster']==1]['Annual Income (k$)'],
            X[data['Cluster']==1]['Spending Score (1-100)'],
            color='green', label='Cluster 1')

plt.scatter(X[data['Cluster']==2]['Annual Income (k$)'],
            X[data['Cluster']==2]['Spending Score (1-100)'],
            color='orange', label='Cluster 2')

plt.scatter(X[data['Cluster']==3]['Annual Income (k$)'],
            X[data['Cluster']==3]['Spending Score (1-100)'],
            color='purple', label='Cluster 3')

plt.scatter(X[data['Cluster']==4]['Annual Income (k$)'],
            X[data['Cluster']==4]['Spending Score (1-100)'],
            color='red', label='Cluster 4')

plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers_[:,1],
            s=200,
            color='black',
            label='Centroids')

plt.title("K-Means Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()
```

## Output:

<img width="1161" height="821" alt="Screenshot 2026-08-29 183310" src="https://github.com/user-attachments/assets/b2392e70-8ad3-4851-b8be-9c356b2936db" />



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
