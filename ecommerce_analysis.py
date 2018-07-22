# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 19:33:22 2018

@author: Abhinav
"""

import os
import pandas as pd
import numpy as np

os.chdir('C:/Users/Abhinav/Downloads/My Assigments/Customer Analytics/Ecommerce_analysis/Ecommerce_analysis')

ecomdata = pd.read_csv('data.csv',encoding='latin1')

ecomdata.info()

ecomdata.describe()

#Converting Invoice Date to date format

ecomdata['InvoiceDate'] = pd.to_datetime(ecomdata['InvoiceDate'],format = "%m/%d/%Y %H:%M")

ecomdata['CustomerID'] = ecomdata['CustomerID'].astype('category')
#extracting Month

ecomdata['Month'] = ecomdata['InvoiceDate'].dt.month

ecomdata['Year'] = ecomdata['InvoiceDate'].dt.year

ecomdata['Day'] = ecomdata['InvoiceDate'].dt.day

ecomdata['Hour'] = ecomdata['InvoiceDate'].dt.hour

ecomdata['Month'] = ecomdata['Month'].astype('category').cat.rename_categories(['Jan','Feb','Mar','Apr','May','June','July','Aug','Sept','Oct','Nov','Dec'])

#Check data for returns

negquantdata = ecomdata[ecomdata['Quantity'] <0]

ecomdata['Total'] = ecomdata['Quantity'] * ecomdata['UnitPrice']

#Summarize the sales by Country, Year, Month

Conttotalsales = pd.DataFrame(ecomdata.groupby('Country')['Total'].sum()).reset_index()

Contyearsales = pd.DataFrame(ecomdata.groupby(['Country','Year'])['Total'].sum()).reset_index()

Contyearmonthsales = pd.DataFrame(ecomdata.groupby(['Country','Year','Month'])['Total'].sum()).reset_index()

monthsales = pd.DataFrame(ecomdata.groupby(['Month'])['Total'].sum()).reset_index()

yearmonthsales = pd.DataFrame(ecomdata.groupby(['Year','Month'])['Total'].sum()).reset_index()

daysales = pd.DataFrame(ecomdata.groupby(['Day'])['Total'].sum()).reset_index()

hoursales = pd.DataFrame(ecomdata.groupby(['Hour'])['Total'].sum()).reset_index()

# Removing returns, negtive price & missing customer ID
ecomdata['CustomerID'].isnull().values.any()

segdata = ecomdata[(ecomdata['Quantity'] >= 0) & (ecomdata['UnitPrice'] >= 0) & (pd.notnull(ecomdata['CustomerID']))]

#Prepraing data for Customer Segmentation

custdata = segdata[['CustomerID','Description']]

custdata['values'] = 1

cust_data = pd.crosstab(index = segdata['CustomerID'],columns =segdata['Description'])

#Using PCA to reduce dimension

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale

#convert it to numpy arrays
X= cust_data.values

#Scaling the values
X = scale(X)

pca = PCA(n_components=3877)

pca.fit(X)

#The amount of variance that each PC explains
var= pca.explained_variance_ratio_

#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

plt.plot(var1)

#Looking at above plot I'm taking 30 variables
pca = PCA(n_components=1000)
pca.fit(X)
X1=pca.fit_transform(X)

from sklearn.cluster import KMeans
import matplotlib.cm as cm

cluster_errors = []


for num_clusters in range(10,16):
  clusters = KMeans( num_clusters )
  clusters.fit( X1 )
  cluster_errors.append( clusters.inertia_ )

clusters_df = pd.DataFrame( { "num_clusters":range(10,16), "cluster_errors": cluster_errors } )

plt.figure(figsize=(12,6))
plt.plot(clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )

#Using Silhoutte method to obtain optimal no of cluster

from sklearn.metrics import silhouette_samples, silhouette_score

cluster_range = range( 10, 16 )

for n_clusters in cluster_range:
  # Create a subplot with 1 row and 2 columns
  fig, (ax1, ax2) = plt.subplots(1, 2)
  fig.set_size_inches(18, 7)

  # The 1st subplot is the silhouette plot
  # The silhouette coefficient can range from -1, 1 but in this example all
  # lie within [-0.1, 1]
  ax1.set_xlim([-0.1, 1])
  # The (n_clusters+1)*10 is for inserting blank space between silhouette
  # plots of individual clusters, to demarcate them clearly.
  ax1.set_ylim([0, len(X1) + (n_clusters + 1) * 10])

  # Initialize the clusterer with n_clusters value and a random generator
  # seed of 10 for reproducibility.
  clusterer = KMeans(n_clusters=n_clusters, random_state=10)
  cluster_labels = clusterer.fit_predict( X1 )

  # The silhouette_score gives the average value for all the samples.
  # This gives a perspective into the density and separation of the formed
  # clusters
  silhouette_avg = silhouette_score(X1, cluster_labels)
  print("For n_clusters =", n_clusters,
        "The average silhouette_score is :", silhouette_avg)

  # Compute the silhouette scores for each sample
  sample_silhouette_values = silhouette_samples(X1, cluster_labels)

  y_lower = 10
  for i in range(n_clusters):
      # Aggregate the silhouette scores for samples belonging to
      # cluster i, and sort them
      ith_cluster_silhouette_values = \
          sample_silhouette_values[cluster_labels == i]

      ith_cluster_silhouette_values.sort()

      size_cluster_i = ith_cluster_silhouette_values.shape[0]
      y_upper = y_lower + size_cluster_i

      color = cm.spectral(float(i) / n_clusters)
      ax1.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)

      # Label the silhouette plots with their cluster numbers at the middle
      ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

      # Compute the new y_lower for next plot
      y_lower = y_upper + 10  # 10 for the 0 samples

  ax1.set_title("The silhouette plot for the various clusters.")
  ax1.set_xlabel("The silhouette coefficient values")
  ax1.set_ylabel("Cluster label")

  # The vertical line for average silhoutte score of all the values
  ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

  ax1.set_yticks([])  # Clear the yaxis labels / ticks
  ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

  # 2nd Plot showing the actual clusters formed
  colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
  ax2.scatter(X1[:, 0], X1[:, 1], marker='.', s=30, lw=0, alpha=0.7,
              c=colors)

  # Labeling the clusters
  centers = clusterer.cluster_centers_
  # Draw white circles at cluster centers
  ax2.scatter(centers[:, 0], centers[:, 1],
              marker='o', c="white", alpha=1, s=200)

  for i, c in enumerate(centers):
      ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

  ax2.set_title("The visualization of the clustered data.")
  ax2.set_xlabel("Feature space for the 1st feature")
  ax2.set_ylabel("Feature space for the 2nd feature")

  plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                "with n_clusters = %d" % n_clusters),
               fontsize=14, fontweight='bold')

  plt.show()


#Based on above findings choosing k=10

kmeans_model = KMeans(n_clusters=10, random_state=1234)

kmeans_model.fit(X1)

clusters = kmeans_model.labels_

#Using kmodes 

from kmodes.kmodes import KModes

km = KModes(n_clusters=10, init='Huang', n_init=5, verbose=1)

clusters = km.fit_predict(X1)

#Using Kprototype
from kmodes.kprototypes import KPrototypes

kproto = KPrototypes(n_clusters=n_clusters, init='Huang', verbose=1)

#Don't execute only with numerical variables
clusters = kproto.fit_predict(X1) #Use ,categorical=[1, 2])

#Market Basket analysis

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

productdata = segdata[['InvoiceNo','Description']]

productdata['values'] = 1

prod_data = pd.crosstab(index = segdata['InvoiceNo'],columns =segdata['Description'])

frequent_itemsets = apriori(prod_data, min_support=0.001, use_colnames=True,max_len = 3)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

rules.head()
