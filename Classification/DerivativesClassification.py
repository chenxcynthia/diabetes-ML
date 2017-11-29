#import libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

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


# Multiclass Classification

# 1 = SSPG value is greater than 150
# 2 = SSPG value is less than or equal to 150.

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
                a = np.append(a,height_values[k])
                a = np.append(a,weight_values[k])
                b = np.append(b,height_values[k])
                b = np.append(b,weight_values[k])
                derivative = b - a
                X.append(derivative)

                start = SSPG_values[k];
                end = SSPG_values[k + 1]
                SSPG_difference = end - start
                
                if SSPG_difference > 0:
                    Y.append(0)
                else:
                    Y.append(1)


# Data preprocessing
X_scaled = preprocessing.scale(X); #feature scaling
X_normalized = preprocessing.normalize(X_scaled); #normalization
X_train, X_test, Y_train, Y_test, = train_test_split(X, Y, test_size=0.2, random_state=15)


# Logistic Regression

from sklearn.linear_model import LogisticRegressionCV

# Create the classifier
classifier = LogisticRegressionCV(Cs=10, fit_intercept=True, penalty='l2', scoring=None, 
            solver='sag', tol=0.0001, verbose=1, refit=True, 
                                  intercept_scaling=1.0, random_state=None)
# Train the classifier 
classifier.fit(X_train, Y_train)
# Predict the topics of the test set and compute the evaluation metrics
y_pred = classifier.predict(X_test)
print(classifier.score(X_test, Y_test))


from sklearn.metrics import confusion_matrix
import itertools

cmatrix = confusion_matrix(Y_test, y_pred)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cmatrix, classes = [1, 2], title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cmatrix,  classes = [1, 2],normalize=True, title='Normalized confusion matrix')
plt.show()

precision_recall_fscore_support(Y_test, y_pred, average = 'macro')

coefficients = classifier.coef_
pd.DataFrame(coefficients)

# Other classifiers

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

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test, = train_test_split(X, Y, test_size=0.2, random_state=15)

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

for name, clf in zip(names, classifiers):
#     ax = plt.subplot(len(X), len(Y) + 1, i)
    
    clf.fit(X_train, Y_train)
    score = clf.score(X_test, Y_test)
    print('Method: ' + name)
    print(score) #'Score: ' + 
    
    y_pred = clf.predict(X_test)
    cmatrix = confusion_matrix(Y_test, y_pred)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cmatrix, classes = [1, 2], title='Confusion matrix, ' + name)
    plt.show()
    i += 1


# #### Cross Validation (for Adaboost and Random Forest)

from sklearn.cross_validation import KFold
import pylab as pl
# compute RMSE using 5-fold x-validation

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA", "Logistic Regression"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()]

k = 5 # number of folds for cross validation

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
#         # Plot non-normalized confusion matrix
#         plt.figure()
#         plot_confusion_matrix(cmatrix, classes = [1, 2], title='Confusion matrix, without normalization')
#     #     # Plot normalized confusion matrix
#     #     plt.figure()
#     #     plot_confusion_matrix(confusion_matrix,  classes = [1, 2],normalize=True, title='Normalized confusion matrix')
#         plt.show()
#         i += 1

    print('Method: ' + name + ' Cross Validation')
    cv_err = total_err / k
    print(cv_err)
    print(cv_matrix)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cv_matrix, classes = [1, 2], title='Confusion matrix, Cross Validation, ' + name)
#     # Plot normalized confusion matrix
#     plt.figure()
#     plot_confusion_matrix(confusion_matrix,  classes = [1, 2],normalize=True, title='Normalized confusion matrix')
    plt.show()
    i += 1

X = np.asarray(X)
Y = np.asarray(Y)

from scipy import stats

X_normalized = np.zeros(X.shape)
for i in range(X.shape[1]):
    X_normalized[:, i] = stats.zscore(X[:, i])

from sklearn.metrics import accuracy_score

for name, clf in zip(names, classifiers):
#     kf = KFold(len(X), n_folds = k, random_state = 15)
    total_score = 0
    total_acc = 0
    cv_matrix = [[0, 0], [0, 0]]
    for i in [0, 1, 2, 3, 4]:
        X_train_cv = X_normalized[0:i*8]
        X_train_cv = np.concatenate((X_train_cv, X_normalized[(i*8+8):38]))
        X_test_cv = X_normalized[i*8 : i*8+8]

        Y_train_cv = Y[0:(i*8)]
        Y_train_cv = np.concatenate((Y_train_cv, Y[(i*8+8):38]))
        Y_test_cv = Y[(i*8) : (i*8+8)]
        
       # X_train.append(X_train())
        clf.fit(X_train_cv, Y_train_cv)

        # p = np.array([linreg.predict(xi) for xi in x[test]])
        score = clf.score(X_test_cv, Y_test_cv)
        # total_score += score

        y_pred = clf.predict(X_test_cv)
        total_acc += accuracy_score(Y_test_cv, y_pred)
#         cmatrix = confusion_matrix(Y_test_cv, y_pred)
#         cv_matrix += cmatrix

    print('Method: ' + name + ' Cross Validation')
    cv_score = total_acc / k
    print(cv_score)

from scipy import stats

X_normalized = np.zeros(X.shape)
for i in range(X.shape[1]):
    X_normalized[:, i] = stats.zscore(X[:, i])

from sklearn.linear_model import LogisticRegressionCV

X_train, X_test, Y_train, Y_test, = train_test_split(X_normalized, Y, test_size=0.2, random_state=15)

# Create the classifier 
classifier = LogisticRegressionCV(Cs=10, fit_intercept=True, penalty='l2', scoring=None, 
            solver='sag', tol=0.0001, verbose=1, refit=True, 
                                  intercept_scaling=1.0, random_state=None)

# Train the classifier 
classifier.fit(X_train, Y_train)

# Predict the topics of the test set and compute the evaluation metrics
y_pred = classifier.predict(X_test)
# print(precision_score(test_labels_pe_acute, y_pred, average='weighted'))
# probs = classifier.predict_proba(X_test_word_average)
# y_pred  = probs[:,1]

print(classifier.score(X_test, Y_test))

h = .02  # step size in the mesh

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

X = np.asarray(X)

rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, Y)

X_train, X_test, Y_train, Y_test, = train_test_split(X, Y, test_size=0.2, random_state=15)

figure = plt.figure(figsize=(27, 9))
i = 1
# preprocess dataset, split into training and test part
X_train, X_test, Y_train, Y_test, = train_test_split(X, Y, test_size=0.2, random_state=15)

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot(len(classifiers) + 1, i)
ax.set_title("Input data")
# Plot the training points
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
# and testing points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())

# iterate over classifiers
for name, clf in zip(names, classifiers):
    ax = plt.subplot(len(classifiers) + 1, i)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
               alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if ds_cnt == 0:
        ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')
    print('Method: ' + name)
    print(score)

plt.tight_layout()
plt.show()

