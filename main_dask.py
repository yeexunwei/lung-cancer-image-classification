"""
Created on Mon Oct  7 23:27:49 2019

@author: hsunwei
"""

import numpy as np
import pandas as pd

import time

# preprocessing
from config import DATA_CSV, PIXEL_ARRAY, DTYPE, Y_LABEL
from config import FEATURES, OUTPUT_FOLDER, FORMAT
from config import FEATURES_ARRAY, FEATURES_ARRAY2, FEATURES_ARRAY3
from preprocessing import to_arr
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler

# pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

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
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, roc_auc_score

import dask
import dask.array as da
from dask_ml.preprocessing import MinMaxScaler
from dask_ml.wrappers import Incremental
from dask_ml.model_selection import train_test_split

# from dask.distributed import Client
# client = Client(n_workers=4, threads_per_worker=1)
# client

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
    ('MLP', MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', random_state=RANDOM_STATE))
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
    # ('to_arr', FunctionTransformer(to_arr)),
    ('minmax', MinMaxScaler()),
    # ('pca', PCA())
])

t0 = time.time()


"Load and split data"

y = np.load(Y_LABEL)


"Find best classifier"

print("finding best classifier")

features = FEATURES_ARRAY

# [OUTPUT_FOLDER + 'lbp' + FORMAT]: #
# for feature in features:
for feature in [OUTPUT_FOLDER + 'lbp' + FORMAT]:  # features:
    print("""
    ----------------------------------
    getting feature: {}
    """.format(feature))
    X = np.load(feature, allow_pickle=True)
    X = to_arr(X)
    np.save('lbp_arr', X)
    X = da.from_array(X, chunks=X.shape)

    X = transformer_pipe.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    classes = da.unique(y_train).compute()

    "One model for all"
    inc = Incremental(SVC(random_state=RANDOM_STATE), scoring='accuracy')
    for _ in range(10):
        inc.partial_fit(X_train, y_train, classes=classes)
        print('Score:', inc.score(X_test, y_test))

    score = inc.score(X_test, y_test)
    print(score)

    np.save('lbp_svm', score)

time_it()