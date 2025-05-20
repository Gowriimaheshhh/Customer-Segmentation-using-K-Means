#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[4]:


st.set_page_config(page_title="Customer Segmentation", layout="centered")

st.title("Customer Segmentation using K-Means")
st.markdown("Upload the customer data to explore segments.")


# In[5]:


# Upload CSV
file = st.file_uploader(r"C:\Users\Hp\Downloads\archive (2)\Mall_Customers.csv", type=["csv"])
if file:
    df = pd.read_csv(file)
    st.write("Preview of data:", df.head())

    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    # Elbow Plot
    wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    fig1, ax1 = plt.subplots()
    ax1.plot(range(1, 11), wcss, marker='o')
    ax1.set_title('Elbow Method')
    ax1.set_xlabel('Clusters')
    ax1.set_ylabel('WCSS')
    st.pyplot(fig1)

    # Clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    # Scatterplot
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set2', ax=ax2)
    ax2.set_title('Customer Segments')
    st.pyplot(fig2)

    st.subheader("Cluster Summary")
    st.write(df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())


# In[ ]:




