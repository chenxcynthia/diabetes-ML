#import libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# Read demographic and metabolomics data
df_mapping = pd.DataFrame()
curr = pd.read_csv('/Users/cynthiachen/Dropbox/Prediabetic/data/mappingfile.csv', sep=',')
df_mapping = df_mapping.append(curr)

df_metabolomics = pd.DataFrame()
curr = pd.read_csv('/Users/cynthiachen/Dropbox/Prediabetic/data/metabolomics.csv', sep=',')
df_metabolomics = df_metabolomics.append(curr)

current_cols = [col for col in df_metabolomics.columns if '1011' in col]
a  = np.array(df_metabolomics[current_cols[0]])
b = a.transpose()
np.append(b,1)
no_patients = len(df_mapping.patient_id.unique())
patient_id = df_mapping.patient_id.unique()

#Create dataset
X = []
Y = []
labels = []
derivatives = []

# We define 2 classes:
# 1: increase in SSPG values
# 2: decrease in SSPG values
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
                a = np.append(a,height_values[k])
                a = np.append(a,weight_values[k])
                derivative = b - a
                 np.append(derivative, bmi_values[k+1]-bmi_values[k])
                 np.append(derivative, height_values[k+1]-bmi_values[k])
                 np.append(derivative, weight_values[k+1]-bmi_values[k])

                X.append(derivative)

                SSPG_difference = SSPG_values[k + 1] - SSPG_values[k]
                if SSPG_difference > 0:
                    labels.append(1)
                else:
                    labels.append(2)
                Y.append(SSPG_difference)


# Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn import preprocessing
new_dim = len(X) - 1 #new_dim is number of reduced features (after dimensionality reduction)

#perform PCA
pca = PCA(n_components = new_dim)
X_pca = pca.fit_transform(X)

# Data preprocessing:
from sklearn import preprocessing
X_scaled = preprocessing.scale(X_pca); #feature scaling
X_normalized = preprocessing.normalize(X_scaled); #normalization


# Create training + testing sets
from sklearn.cross_validation import train_test_split
# split the data into testing and training data
data_train, data_test, labels_train, labels_test, = train_test_split(X_normalized, Y, test_size=0.2, random_state=15)



# Linear Regression

from sklearn.linear_model import LinearRegression

#Create linear regression object
linreg = LinearRegression()

# # Train the model using the training sets
linreg.fit(data_train, labels_train)

# Compute RMSE on training data
# p = np.array([linreg.predict(xi) for xi in x])
# Now we can constuct a vector of errors
p = linreg.predict(data_train)
err = abs(p - labels_train)
# Dot product of error vector with itself gives us the sum of squared errors
total_error = np.dot(err,err)
# Compute RMSE
rmse = np.sqrt(total_error/len(p))
print('RMSE for training data: %.4f' %rmse)
print (rmse)
print('Variance score: %.2f' % linreg.score(data_train, labels_train))

import pylab as pl
# Plot outputs
get_ipython().magic(u'matplotlib inline')
pl.plot(p, labels_train,'ro')
pl.plot([-75,75],[-75,75], 'g-')
pl.xlabel('predicted')
pl.ylabel('real')
pl.show()

# Compute RMSE on testing data
# p = np.array([linreg.predict(xi) for xi in x])
# constuct a vector of errors
p = linreg.predict(data_test)
err = abs(p - labels_test)
# Dot product of error vector with itself gives the sum of squared errors
total_error = np.dot(err,err)
rmse = np.sqrt(total_error/len(p))

# Print model's coefficients
print('Coefficients: ', linreg.coef_)
# The root mean squared error (RMSE)
print("Root mean squared error: %.2f"
      % rmse)
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % linreg.score(data_test, labels_test))


# Cross validation for Linear regression:

from sklearn.cross_validation import KFold
import pylab as pl
# compute RMSE using 5-fold x-validation

kf = KFold(len(X), n_folds=5)
xval_err = 0
for train, test in kf:
    print("TRAIN:", train, "TEST:", test)
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    
    for i in train:
        X_train.append(X[i])
        Y_train.append(Y[i])
    
    for i in test:
        X_test.append(X[i])
        Y_test.append(Y[i])
    
    linreg.fit(X_train, Y_train)
    p = linreg.predict(X_test)
    e = p-Y_test
    xval_err += np.dot(e,e)

    # Plot outputs
    get_ipython().magic(u'matplotlib inline')
    pl.plot(p, Y_test,'ro')
    pl.plot([-75,75],[-75,75], 'g-')
    pl.xlabel('predicted')
    pl.ylabel('real')
    pl.show()
    
rmse_10cv = np.sqrt(xval_err/len(X_train))
method_name = 'Linear Regression'
print('Method: %s' %method_name)
print('Coefficients: ', linreg.coef_)
print('RMSE without cross validation: %.4f' %rmse)
print('RMSE on 5-fold cross validation: %.4f' %rmse_10cv)
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % linreg.score(data_test, labels_test))


# Ridge Regression

# split the data into testing and training data
data_train, data_test, labels_train, labels_test, = train_test_split(X, Y, test_size=0.2, random_state=15)

from sklearn.linear_model import Ridge
# Create linear regression object with a ridge coefficient 0.5
ridge = Ridge(fit_intercept=True, alpha=0.01)
# Train the model using the training set
ridge.fit(data_train, labels_train)

p = ridge.predict(data_train)
err = p-labels_train
total_error = np.dot(err,err)
rmse_train = np.sqrt(total_error/len(p))

# Compute RMSE using 5-fold x-validation
kf = KFold(len(X), n_folds=5)
xval_err = 0
for train,test in kf:
    print("TRAIN:", train, "TEST:", test)
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    
    for i in train:
        X_train.append(X[i])
        Y_train.append(Y[i])
    
    for i in test:
        X_test.append(X[i])
        Y_test.append(Y[i])
    
    ridge.fit(X_train, Y_train)
    p = ridge.predict(X_test)
    e = p-Y_test
    xval_err += np.dot(e,e)

    # Plot outputs
    get_ipython().magic(u'matplotlib inline')
    pl.plot(p, Y_test,'ro')
    pl.plot([-75,75],[-75,75], 'g-')
    pl.xlabel('predicted')
    pl.ylabel('real')
    pl.show()
    
rmse_10cv = np.sqrt(xval_err/len(X))
method_name = 'Ridge Regression'
print('Method: %s' %method_name)
print('Coefficients: ', ridge.coef_)
print('RMSE on training: %.4f' %rmse_train)
print('RMSE on 5-fold cross validation: %.4f' %rmse_10cv)
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % ridge.score(data_test, labels_test))


# LASSO

# split the data into testing and training data
data_train, data_test, labels_train, labels_test, = train_test_split(X, Y, test_size=0.2, random_state=15)

from sklearn.linear_model import Lasso
# Create Lasso object
lasso = Lasso(fit_intercept=True, alpha=0.01) #alpha is a hyperparameter
# Train the model using the training set
lasso.fit(data_train, labels_train)

p = lasso.predict(data_train)
err = p-labels_train
total_error = np.dot(err,err)
rmse_train = np.sqrt(total_error/len(p))

# Compute RMSE using 5-fold x-validation
kf = KFold(len(X), n_folds=5)
xval_err = 0
for train,test in kf:
    print("TRAIN:", train, "TEST:", test)
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    
    for i in train:
        X_train.append(X[i])
        Y_train.append(Y[i])
    
    for i in test:
        X_test.append(X[i])
        Y_test.append(Y[i])
    
    lasso.fit(X_train, Y_train)
    p = lasso.predict(X_test)
    e = p-Y_test
    xval_err += np.dot(e,e)

    # Plot outputs
    get_ipython().magic(u'matplotlib inline')
    pl.plot(p, Y_test,'ro')
    pl.plot([-75,75],[-75,75], 'g-')
    pl.xlabel('predicted')
    pl.ylabel('real')
    pl.show()
    
rmse_10cv = np.sqrt(xval_err/len(X))
method_name = 'Lasso'
print('Method: %s' %method_name)
print('Coefficients: ', lasso.coef_)
print('RMSE on training: %.4f' %rmse_train)
print('RMSE on 5-fold cross validation: %.4f' %rmse_10cv)
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % lasso.score(data_test, labels_test))

# The coefficients
print('Coefficients: ', lasso.coef_)
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % lasso.score(data_test, labels_test))

# LassoCV

from sklearn.linear_model import LassoCV
lasso = LassoCV(fit_intercept=True, cv = 5)
lasso.fit(data_train, labels_train)

p = lasso.predict(data_train)
err = p-labels_train
total_error = np.dot(err,err)
rmse_train = np.sqrt(total_error/len(p))

# Compute RMSE using 5-fold x-validation
kf = KFold(len(X), n_folds=5)
xval_err = 0
for train,test in kf:
    print("TRAIN:", train, "TEST:", test)
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    
    for i in train:
        X_train.append(X[i])
        Y_train.append(Y[i])
    
    for i in test:
        X_test.append(X[i])
        Y_test.append(Y[i])
    
    lasso.fit(X_train, Y_train)
    p = lasso.predict(X_test)
    e = p-Y_test
    xval_err += np.dot(e,e)

    # Plot outputs
    get_ipython().magic(u'matplotlib inline')
    pl.plot(p, Y_test,'ro')
    pl.plot([-75,75],[-75,75], 'g-')
    pl.xlabel('predicted')
    pl.ylabel('real')
    pl.show()
    
rmse_10cv = np.sqrt(xval_err/len(X))
method_name = 'Lasso'
print('Method: %s' %method_name)
print('Coefficients: ', lasso.coef_)
print('RMSE on training: %.4f' %rmse_train)
print('RMSE on 5-fold cross validation: %.4f' %rmse_10cv)
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % lasso.score(data_test, labels_test))


# Elastic Net

from sklearn.linear_model import ElasticNet
data_train, data_test, labels_train, labels_test, = train_test_split(X, Y, test_size=0.2, random_state=15)
elastic_net = ElasticNet(fit_intercept=True, alpha = 0.001) 
elastic_net.fit(data_train, labels_train)

p = elastic_net.predict(data_train)
err = p-labels_train
total_error = np.dot(err,err)
rmse_train = np.sqrt(total_error/len(p))

# Compute RMSE using 5-fold x-validation
kf = KFold(len(X), n_folds=5)
xval_err = 0
for train,test in kf:
    print("TRAIN:", train, "TEST:", test)
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    
    for i in train:
        X_train.append(X[i])
        Y_train.append(Y[i])
    
    for i in test:
        X_test.append(X[i])
        Y_test.append(Y[i])
    
    elastic_net.fit(X_train, Y_train)
    p = elastic_net.predict(X_test)
    e = p-Y_test
    xval_err += np.dot(e,e)

    # Plot outputs
    get_ipython().magic(u'matplotlib inline')
    pl.plot(p, Y_test,'ro')
    pl.plot([-75,75],[-75,75], 'g-')
    pl.xlabel('predicted')
    pl.ylabel('real')
    pl.show()
    
rmse_10cv = np.sqrt(xval_err/len(X))
method_name = 'Elastic Net'
print('Method: %s' %method_name)
print('Coefficients: ', elastic_net.coef_)
print('RMSE on training: %.4f' %rmse_train)
print('RMSE on 5-fold cross validation: %.4f' %rmse_10cv)
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % elastic_net.score(data_test, labels_test))

