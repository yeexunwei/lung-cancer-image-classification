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
from config import FEATURES_ARRAY, FEATURES_ARRAY2, FEATURES_ARRAY3
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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier  # , GradientBoostingClassifier
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

RANDOM_STATE = 0

classifier = DecisionTreeClassifier(random_state=RANDOM_STATE)
classifiers = [
    # ('NB', GaussianNB()),
    # ('LSVC', LinearSVC()),
    ('SVC', SVC(random_state=RANDOM_STATE, gamma='scale')),
    ('LOGR', LogisticRegression(random_state=RANDOM_STATE, solver='lbfgs', max_iter=100)),
    # ('SGD', SGDClassifier()),
    # ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier(criterion='gini', random_state=RANDOM_STATE)),
    ('RF', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    # ('ABC', AdaBoostClassifier()),
    # ('XGB', xgb.XGBRegressor()),
    # ('GBC', GradientBoostingClassifier()),
    # ('LDA', LinearDiscriminantAnalysis()),
    ('MLP', MLPClassifier())
]


def time_it():
    dif = (time.time() - t0) / 60
    print("Used time: {:.2f}".format(dif))


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


def save_result(all_results, filename):
    compare = pd.DataFrame(all_results)
    compare['feature'] = features
    compare = compare.set_index('feature')
    compare.to_pickle(filename + '.pkl')


"Pipeline"

transformer_pipe = Pipeline(steps=[
    ('to_arr', FunctionTransformer(to_arr)),
    ('minmax', MinMaxScaler()),
    # ('pca', PCA())
])

# extraction_transformer = Pipeline(steps=[
#     #     ('cluster', KMeans()),
# ])
#
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('trans', transformer_pipe, TRANSFORMATION_LIST)
#     ])

t0 = time.time()

"Load and split data"

# print("loading data")
#
# X = np.load(OUTPUT_FOLDER + 'll.npy', allow_pickle=True)
y = np.load(Y_LABEL)

# time_it()


"Find best classifier"

print("finding best classifier")

features = FEATURES_ARRAY
all_acc = []
all_rec = []

for feature in features:
    print("""
    ----------------------------------
    getting feature {}: {}
    """.format(features.index(feature), feature))
    X = np.load(feature, allow_pickle=True)

    X = transformer_pipe.fit_transform(X)

    # RandomOverSampler(random_state=RANDOM_STATE)
    # ADASYN(random_state=RANDOM_STATE)
    smote = SMOTE(random_state=RANDOM_STATE)
    smote.fit(X, y)
    X, y = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    feature_acc = []
    feature_rec = []
    for name, classifier in classifiers:
        pipe = Pipeline(steps=[
            ('classifier', classifier)
        ])
        pipe.fit(X_train, y_train)

        # score = pipe.score(X_test, y_test)
        acc_score = cross_val_score(pipe, X_train, y_train, cv=10, scoring='accuracy')
        acc_result = {name: acc_score}

        rec_score = cross_val_score(pipe, X_train, y_train, cv=10, scoring='recall')
        rec_result = {name: rec_score}

        feature_acc.append(acc_result)
        feature_rec.append(rec_result)
        print(acc_score, rec_score)

    all_acc.append(feature_acc)
    all_rec.append(feature_rec)


save_result(all_acc, 'accuracy')
save_result(all_rec, 'recall')

time_it()

# "Cross validation to select best feature"
#
# pipeline = Pipeline(steps=[
#     ('transform', transformer_pipe),
#     ('classifier', DecisionTreeClassifier(random_state=9))
# ])
#
# scores = cross_val_score(pipeline,X_train,y_train,cv=10,scoring='accuracy')
# print(scores)
#
#


#
# "Hyperparam tuning"
#
# param_dist = {"max_depth": [3, None],
#               "max_features": [1, None],
#               "min_samples_split": sp_randint(2, 11),
#               "criterion": ["gini", "entropy"]}
#
# n_iter_search = 5
# random_search = RandomizedSearchCV(classifier, param_distributions=param_dist,
#                                    n_iter=n_iter_search, cv=10, iid=False)
#
# random_search.fit(X_train, y_train)
# report(random_search.cv_results_)
