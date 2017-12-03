import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import random
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

filename = '../Combined Trajectory_Label_Geolife/Hand_Crafted_features.csv'
np.random.seed(7)
random.seed(7)

df = pd.read_csv(filename)
X = np.array(df.loc[:, df.columns != 'Label'])
Y = np.array(df['Label'])

# Split Data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=7)

# Random forest Grid Search
RandomForest = RandomForestClassifier()
parameters = {'n_estimators': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]}
clf = GridSearchCV(estimator=RandomForest, param_grid=parameters, cv=5)
fit = clf.fit(X_train, y_train)
print('optimal parameter value: ', fit.best_params_)
Prediction_RT = fit.best_estimator_.predict(X_test)
Accuracy_RandomForest = len(np.where(Prediction_RT == y_test)[0]) * 1. / len(y_test)
print('Accuracy: ', Accuracy_RandomForest)
print(classification_report(y_test, Prediction_RT, digits=3))

# Multilayer perceptron
MLP = MLPClassifier(early_stopping=True, hidden_layer_sizes=(2 * np.shape(X_train)[1],))
parameters = {'hidden_layer_sizes': [(2 * np.shape(X_train)[1],)]}
clf = GridSearchCV(estimator=MLP, param_grid=parameters, cv=5)
fit = clf.fit(X_train, y_train)
print('optimal parameter value: ', fit.best_params_)
Prediction_MLP = fit.best_estimator_.predict(X_test)
Accuracy_MLP = len(np.where(Prediction_MLP == y_test)[0]) * 1. / len(y_test)
print('Accuracy: ', Accuracy_MLP)
print(classification_report(y_test, Prediction_MLP, digits=3))

# Decision Tree Grid Search
DT = DecisionTreeClassifier()
parameters = {'max_depth': [1, 5, 10, 15, 20, 25, 30, 35, 40]}
clf = GridSearchCV(estimator=DT, param_grid=parameters, cv=5)
fit = clf.fit(X_train, y_train)
print('optimal parameter value: ', fit.best_params_)
Prediction_DT = fit.best_estimator_.predict(X_test)
Accuracy_DecisionTree = len(np.where(Prediction_DT == y_test)[0]) * 1. / len(y_test)
print('Accuracy: ', Accuracy_DecisionTree)
print(classification_report(y_test, Prediction_DT, digits=3))

# SVM Grid Search
SVM = SVC()
parameters = {'C': [0.5, 1, 4, 7, 10, 13, 16, 20]}
clf = GridSearchCV(estimator=SVM, param_grid=parameters, cv=5)
fit = clf.fit(X_train, y_train)
print('optimal parameter value: ', fit.best_params_)
Prediction_SVM = fit.best_estimator_.predict(X_test)
Accuracy_SVM = len(np.where(Prediction_SVM == y_test)[0]) * 1. / len(y_test)
print('Accuracy: ', Accuracy_SVM)
print(classification_report(y_test, Prediction_SVM, digits=3))

# KNN Grid Search
KNN = KNeighborsClassifier()
parameters = {'n_neighbors': [3, 5, 10, 15, 20, 25, 30, 35, 40]}
clf = GridSearchCV(estimator=KNN, param_grid=parameters, cv=5)
fit = clf.fit(X_train, y_train)
print('optimal parameter value: ', fit.best_params_)
Prediction_KNN = fit.best_estimator_.predict(X_test)
Accuracy_KNN = len(np.where(Prediction_KNN == y_test)[0]) * 1. / len(y_test)
print('Accuracy: ', Accuracy_KNN)
print(classification_report(y_test, Prediction_KNN, digits=3))



