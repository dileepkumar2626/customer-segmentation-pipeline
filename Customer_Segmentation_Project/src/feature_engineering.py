import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.decomposition import PCA

pre_processed_data=pd.read_csv("data/processed/pre_processed_data.csv")

def encoding(pre_processed_data):
    num_features=pre_processed_data.select_dtypes("number")
    cols=["Gender","Ever_Married","Graduated","Profession","Spending_Score","Var_1"]
    featured_data=pd.get_dummies(pre_processed_data[cols],drop_first=True)
    featured_data=featured_data.astype(int)
    encoded_data= pd.concat([featured_data, num_features], axis=1)
    return encoded_data
def data_export(data):
    return data.to_csv("data/processed/encoded_data.csv",index=False)
def scaling(data):
    scale=StandardScaler()
    scaled_data=scale.fit_transform(data)
    return scaled_data
def pca(scaled_data):
    pca_model=PCA(n_components=2)
    reduced_scaled_data=pca_model.fit_transform(scaled_data)
    return reduced_scaled_data

encoded_data=encoding(pre_processed_data)
print(encoded_data)
data_export(encoded_data)
scaled_data=scaling(encoded_data)
reduced_scaled_data=pca(scaled_data)
print(reduced_scaled_data)
np.savetxt("data/processed/pca_data.txt",reduced_scaled_data)