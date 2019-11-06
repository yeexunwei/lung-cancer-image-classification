#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:27:16 2019

@author: hsunwei
"""


from config import SCORING
from preprocessing import to_arr, mms_ft, pca_ft, label_ft


from sklearn.preprocessing import LabelEncoder
import numpy as np
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
# models.append(('LiR', LinearRegression()))
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


# run for each attribute in df
def run_model(df, y, mms=False, pca=False):
    result_list = []
    for col in df.columns:
        X = df[col]
        X = to_arr(X)
        if mms:
            X = mms_ft(X)
        if pca:
            X = pca_ft(X)

        result = {}
        for name, model in models:
            cv_results = cross_validate(model, X, y, cv=10, scoring=scoring)
            result[name] = cv_results

        result_list.append(result)
    return result_list


#
def run_desc_model(histo_lists, y):
    result_list = []
    for histo_list in histo_lists:
        X = np.array(histo_list)

        result = {}
        for name, model in models:
            cv_results = cross_validate(model, X, y, cv=10, scoring=scoring)
            result[name] = cv_results

        result_list.append(result)
    return result_list

#%%
