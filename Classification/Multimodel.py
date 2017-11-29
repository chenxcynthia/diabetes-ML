#import libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.cross_validation import KFold
import pylab as pl
from scipy import stats

from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV


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
    weight_values = df_mapping.loc[df_mapping.patient_id == curr_id].WEIGHT
    weight_values = np.asarray(weight_values)
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
                X.append(derivative)

                start = SSPG_values[k];
                end = SSPG_values[k + 1]
                SSPG_difference = end - start
                
                if SSPG_difference > 0:
                    Y.append(0)
                else:
                    Y.append(1)
           
# #data preprocessing
# X_scaled = preprocessing.scale(X); #feature scaling
# X_normalized = preprocessing.normalize(X_scaled); #normalization

X_train, X_test, Y_train, Y_test, = train_test_split(X, Y, test_size=0.2, random_state=15)

# Create the classifier 
classifier = LogisticRegressionCV()

# Train the classifier 
classifier.fit(X_train, Y_train)

# Predict the topics of the test set and compute the evaluation metrics
y_pred = classifier.predict(X_test)

print(classifier.score(X_test, Y_test))


#CONFUSION MATRIX

#Function for printing and plotting confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#creating confusion matrix
cmatrix = confusion_matrix(Y_test, y_pred)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cmatrix, classes = [1, 2], title='Confusion matrix')
plt.show()


# #### Cross Validation
names = ["QDA"]
classifiers = [LogisticRegressionCV()]
k = 5 # number of folds for cross validation

# using 5-fold x-validation
for name, clf in zip(names, classifiers):
    kf = KFold(len(X), n_folds = k, random_state = 15)
    total_err = 0
    cv_matrix = [[0, 0], [0, 0]]
    for train, test in kf:
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
        clf.fit(X_train, Y_train)

        # p = np.array([linreg.predict(xi) for xi in x[test]])
        score = clf.score(X_test, Y_test)
        total_err += score

        y_pred = clf.predict(X_test)
        cmatrix = confusion_matrix(Y_test, y_pred)
        cv_matrix += cmatrix

    print('Method: ' + name + ' Cross Validation')
    cv_err = total_err / k
    print(cv_err)
    print(cv_matrix)
    # Plot confusion matrix
    plt.figure()
    plot_confusion_matrix(cv_matrix, classes = [1, 2], title='Confusion matrix, Cross Validation, ' + name)
    plt.show()
    i += 1

X = np.asarray(X)
Y = np.asarray(Y)
X_normalized = np.zeros(X.shape)
for i in range(X.shape[1]):
    X_normalized[:, i] = stats.zscore(X[:, i])

for name, clf in zip(names, classifiers):
    kf = KFold(len(X), n_folds = k, random_state = 15)
    total_err = 0
    cv_matrix = [[0, 0], [0, 0]]
    for train, test in kf:
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
        clf.fit(X_train, Y_train)

        # p = np.array([linreg.predict(xi) for xi in x[test]])
        score = clf.score(X_test, Y_test)
        total_err += score

        y_pred = clf.predict(X_test)
        cmatrix = confusion_matrix(Y_test, y_pred)
        cv_matrix += cmatrix

    print('Method: ' + name + ' Cross Validation')
    cv_err = total_err / k
    print(cv_err)
    print(cv_matrix)
    # Plot confusion matrix
    plt.figure()
    plot_confusion_matrix(cv_matrix, classes = [1, 2], title='Confusion matrix, Cross Validation, ' + name)
    plt.show()
    i += 1



