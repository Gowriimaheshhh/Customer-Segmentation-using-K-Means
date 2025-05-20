#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# # data collection and analysis

# In[6]:


data= pd.read_csv(r"C:\Users\Hp\Downloads\archive (2)\Mall_Customers.csv")


# In[7]:


#1st 5 rows in DF
data.head()


# In[9]:


#finding no.s rows and columns
data.shape


# In[10]:


#getting some info about DF
data.info()


# In[11]:


#for missing values
data.isnull().sum()


# In[13]:


#choosing annual income column and spending score column
x= data.iloc[:,[3,4]].values
x


# In[15]:


#choosing no. of clustors
#wcss- within clusters sum of squares
#finding wcss values for different clusters
wcss= []
for i in range (1,11):
    kmeans= KMeans(n_clusters=i, init='k-means++',random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[16]:


#plot elbow graph to find which cluster has min vlaue
#elbow point graph or cut-off point graph
sns.set()
plt.plot(range(1,11),wcss)
plt.title("Elbow Point Graph")
plt.xlabel("no. of clusters")
plt.ylabel("WCSS")
plt.show


# In[17]:


#optimum no. of clusters = 5
kmeans= KMeans(n_clusters=5, init='k-means++',random_state=0)

#return a label for each data point based on there cluster
y= kmeans.fit_predict(x)
y


# In[27]:


#visualizing all the clusters
plt.figure(figsize=(10,8))
plt.scatter(x[y==0,0], x[y==0,1], s=50 , c="green", label="cluster 1 ")
plt.scatter(x[y==1,0], x[y==1,1], s=50 , c="blue", label="cluster 2 ")
plt.scatter(x[y==2,0], x[y==2,1], s=50 , c="yellow", label="cluster 3 ")
plt.scatter(x[y==3,0], x[y==3,1], s=50 , c="red", label="cluster 4 ")
plt.scatter(x[y==4,0], x[y==4,1], s=50 , c="violet", label="cluster 5 ")

#plot the centriod
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100, c='black', label='Centriods')
plt.title("Customer Group")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()


# In[ ]:




