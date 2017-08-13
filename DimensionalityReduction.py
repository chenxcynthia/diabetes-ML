#import libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import math

# Read demographic and metabolomics data

df_mapping = pd.DataFrame()
curr = pd.read_csv('/Users/cynthiachen/Dropbox/Prediabetic/data/mappingfile.csv', sep=',')
df_mapping = df_mapping.append(curr)

df_metabolomics = pd.DataFrame()
curr = pd.read_csv('/Users/cynthiachen/Dropbox/Prediabetic/data/metabolomics.csv', sep=',')
df_metabolomics = df_metabolomics.append(curr)

# Create dataset

X = []
Y = []
labels = []
derivatives = []

for i in range(no_patients):
    curr_id = patient_id[i]
    current_cols = [col for col in df_metabolomics.columns if str(curr_id) in col]
    SSPG_values = df_mapping.loc[df_mapping.patient_id == curr_id].SSPG
    SSPG_values = np.asarray(SSPG_values)
    height_values = df_mapping.loc[df_mapping.patient_id == curr_id].height
    height_values = np.asarray(height_values)
    weight_values = df_mapping.loc[df_mapping.patient_id == curr_id].BMI
    weight_values = np.asarray(weight_values)
    bmi_values = df_mapping.loc[df_mapping.patient_id == curr_id].BMI
    bmi_values = np.asarray(bmi_values)
    if not current_cols:
        print("Cannot find the patient",i)
    else:
        for k in range(len(current_cols) - 1):
            if not (math.isnan(SSPG_values[k]) or math.isnan(SSPG_values[k + 1])):
                a  = np.array(df_metabolomics[current_cols[k]])
                b = np.array(df_metabolomics[current_cols[k + 1]])
                a = a.transpose()
                b = b.transpose()
                #a = np.append(a,height_values[k])
                #a = np.append(a,weight_values[k])
                derivative = b - a
                #                 np.append(derivative, bmi_values[k+1]-bmi_values[k])
                #                 np.append(derivative, height_values[k+1]-bmi_values[k])
                #                 np.append(derivative, weight_values[k+1]-bmi_values[k])
                
                X.append(derivative)
                
                SSPG_difference = SSPG_values[k + 1] - SSPG_values[k]
                if SSPG_difference > 0:
                    labels.append(1)
                else:
                    labels.append(2)
                Y.append(SSPG_difference)

# Perform dimensionality reduction on dataset

new_dim = len(X) - 1 #new_dim is number of reduced features (after dimensionality reduction)

#perform PCA
pca = PCA(n_components = new_dim)
X_pca = pca.fit_transform(X)

