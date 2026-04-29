# Libraries
import pandas as pd
import numpy as np
import pathlib as Path

raw_data=pd.read_csv("data/raw/Customer_Segmentation.csv")

print(raw_data.isnull().sum())
print(raw_data["Segmentation"].value_counts())
label_data=raw_data["Segmentation"]
label_data.to_csv("data/processed/label_data.csv",index=False)
# Functions
def cat_feature(raw_data):
    cat_features=raw_data.select_dtypes("object")
    return cat_features
def num_feature(raw_data):
    num_features=raw_data.select_dtypes("number")
    return num_features
def cat_null_handling(cat_features):
    for col in cat_features.columns:
        null_count=((cat_features[col].isnull().sum())/len(cat_features[col]))*100
        if null_count>50:
            cat_features[col]=cat_features.drop(columns=[col])
        elif (null_count<50) and (null_count>0):
            cat_mode=cat_features[col].mode()[0]
            cat_features[col]=cat_features[col].fillna(cat_mode)
        else:
            continue
    return cat_features
def num_null_handling(num_features):
    for col in num_features.columns:
        null_count=((num_features[col].isnull().sum())/len(num_features[col]))*100
        if null_count>50:
            num_features[col]=num_features.drop(columns=[col])
        elif (null_count<50) and (null_count>0):
            num_mode=num_features[col].mode()[0]
            num_features[col]=num_features[col].fillna(num_mode)
        else:
            continue
    return num_features
def exporting_data(data):
    return  data.to_csv("data/processed/pre_processed_data.csv",index=False)

# Function Calling
cat_features=cat_feature(raw_data)
num_features=num_feature(raw_data)
processed_cat_features=cat_null_handling(cat_features)
processed_num_features=num_null_handling(num_features)
pre_processed_data=pd.concat([processed_cat_features,processed_num_features],axis=1)
# As we have Unsupervised Learning Model So we will remove the output "Segmentation" Coulumn and also the Unnamed,ID Columns"
pre_processed_data=pre_processed_data.drop(["Segmentation","Unnamed: 0","ID"],axis=1)
print(pre_processed_data.isnull().sum())
print(pre_processed_data)
exporting_data(pre_processed_data)