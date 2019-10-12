#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:27:16 2019

@author: hsunwei
"""

from config import SCORING
from preprocessing import to_arr, mms_ft, pca_ft, label_ft


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier


from sklearn.metrics import accuracy_score, precision_score, roc_curve, roc_auc_score, classification_report


# Model

# create all the machine learning models
models = []
models.append(('NB', GaussianNB()))
models.append(('LSVC', LinearSVC()))
models.append(('SVC', SVC(random_state=9, gamma='scale')))
models.append(('LiR', LinearRegression()))
models.append(('LoR', LogisticRegression(random_state=9, solver='lbfgs')))
models.append(('SGD', SGDClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=9)))
models.append(('RF', RandomForestClassifier(n_estimators=100, random_state=9)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('MLP', MLPClassifier()))

# variables to hold the results and names
results = []
names = []
scoring = SCORING


def run_model(data, all_cols, mms=False, pca=False):
    y = data['diagnosis']
    y = label_ft(y)

    for col in all_cols:
        print(col)
        X = data[col]
        X = to_arr(X)
        if mms:
            X = mms_ft(X)
        if pca:
            X = pca_ft(X)

        feature = {}
        for name, model in models:
            cv_results = cross_validate(model, X, y, cv=10, scoring=scoring)
            feature[name] = cv_results.mean()

        grid = []
        grid.append(col)

        return grid


#%%
