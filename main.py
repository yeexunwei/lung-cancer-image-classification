"""
Created on Mon Oct  7 23:27:49 2019

@author: hsunwei
"""

# https://github.com/WillKoehrsen/machine-learning-project-walkthrough/blob/master/Machine%20Learning%20Project%20Part%201.ipynb
# https://github.com/WillKoehrsen/machine-learning-project-walkthrough

import numpy as np
import pandas as pd
import pickle
import cv2
import matplotlib.pyplot as plt

import time
from sys import getsizeof

# preprocessing
from config import DATA_CSV, PIXEL_ARRAY, DTYPE, Y_LABEL
from config import FEATURES, OUTPUT_FOLDER, FORMAT
from preprocessing import to_arr
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler

# image processing
from image_processing import sift_ext, surf_ext, orb_ext
from image_processing import sift_des, surf_des, orb_des
from image_processing import generate_bag

# pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from config import TRANSFORMATION_LIST, EXTRACTION_LIST
from sklearn.preprocessing import FunctionTransformer

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.metrics import accuracy_score, log_loss

# model
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

RANDOM_STATE = 9

classifier = DecisionTreeClassifier(random_state=RANDOM_STATE)
classifiers = [
    ('NB', GaussianNB()),
    ('LSVC', LinearSVC()),
    ('SVC', SVC(random_state=RANDOM_STATE, gamma='scale')),
    ('LOGR', LogisticRegression(random_state=RANDOM_STATE, solver='lbfgs')),
    ('SGD', SGDClassifier()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ('RF', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ('ABC', AdaBoostClassifier()),
    ('GBC', GradientBoostingClassifier()),
    ('LDA', LinearDiscriminantAnalysis()),
    ('MLP', MLPClassifier())
]


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


"Pipeline"

transformation_transformer = Pipeline(steps=[
    ('to_arr', FunctionTransformer(to_arr)),
    ('minmax', MinMaxScaler()),
    ('pca', PCA())
])

extraction_transformer = Pipeline(steps=[
    #     ('cluster', KMeans()),
])

preprocessor = ColumnTransformer(
    transformers=[
        ('trans', transformation_transformer, TRANSFORMATION_LIST)
    ])


"Load and split data"

X = np.load(OUTPUT_FOLDER + 'll.npy', allow_pickle=True)
y = np.load(Y_LABEL)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)


"Find best classifier"

result_feature = []

for name, classifier in classifiers:
    pipe = Pipeline(steps=[
#         ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)

    result = {}
    result[name] = score
    print(result)
#     print(classifier)
#     print("model score: %.3f" % score)


"Cross validation"

pipeline = Pipeline(steps=[
    ('transform', transformation_transformer),
    ('classifier', DecisionTreeClassifier(random_state=9))
])

scores = cross_val_score(pipeline,X_train,y_train,cv=10,scoring='accuracy')
print(scores)


"Find best feature"

"Hyperparam tuning"

param_dist = {"max_depth": [3, None],
              "max_features": [1, None],
              "min_samples_split": sp_randint(2, 11),
              "criterion": ["gini", "entropy"]}

n_iter_search = 5
random_search = RandomizedSearchCV(classifier, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=10, iid=False)

random_search.fit(X_train, y_train)
report(random_search.cv_results_)
