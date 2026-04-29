import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from kneed import KneeLocator

pca_scaled_data=np.loadtxt("data/processed/pca_data.txt")

def dbscan(pca_scaled_data):
    model_1=DBSCAN(eps=0.21,min_samples=5)
    dbscan_clustering=model_1.fit_predict(pca_scaled_data)
    return dbscan_clustering
def kmeans(pca_scaled_data):
    wcss=[]
    for i in range(1,11):
        model=KMeans(n_clusters=i,random_state=32)
        kmeans=model.fit_predict(pca_scaled_data)
        wcss.append(model.inertia_)
    knee = KneeLocator(range(1, 11), wcss, curve="convex", direction="decreasing")
    optimal_k = knee.elbow
    return optimal_k
def kmeans_clustering(pca_scaled_data):
    model=KMeans(n_clusters=4)
    kmean_clustering=model.fit_predict(pca_scaled_data)
    return kmean_clustering

dbscan_clustering=dbscan(pca_scaled_data)
optimal_k=kmeans(pca_scaled_data)
print("Best K :",optimal_k)
kmean_clustering=kmeans_clustering(pca_scaled_data)

# Grpah
sns.scatterplot(x=pca_scaled_data[:, 0], y=pca_scaled_data[:,1], c=dbscan_clustering)
plt.show()
sns.scatterplot(x=pca_scaled_data[:, 0], y=pca_scaled_data[:,1], c=kmean_clustering)
plt.show()