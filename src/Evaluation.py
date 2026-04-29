import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
import pickle

pca_scaled_data=np.loadtxt("data/processed/pca_data.txt")
pre_processed_data=pd.read_csv("data/processed/pre_processed_data.csv")

def kmeans_clustering(pca_scaled_data,random_state=42):
    model=KMeans(n_clusters=4)
    k_model=model.fit_predict(pca_scaled_data)
    print(silhouette_score(pca_scaled_data,k_model))
    return k_model
k_model=kmeans_clustering(pca_scaled_data)
pre_processed_data["label"]=k_model
pal = ["red", "blue", "yellow", "green"]
sns.countplot(data=pre_processed_data,x="label",palette=pal)
sns.scatterplot(x=pre_processed_data["Age"],y=pre_processed_data["Ever_Married"],hue=pre_processed_data["label"],palette=pal)
# Based On Graph those persons who are middle age but these are unmarried and those who are married are older.
with open("models/kmeans_model.pkl", "wb") as f:
    pickle.dump(k_model, f)