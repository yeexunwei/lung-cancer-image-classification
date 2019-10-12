#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 23:27:49 2019

@author: hsunwei
"""

# https://github.com/WillKoehrsen/machine-learning-project-walkthrough/blob/master/Machine%20Learning%20Project%20Part%201.ipynb
# https://github.com/WillKoehrsen/machine-learning-project-walkthrough


from config import DATA_DF  # , SCORING
from import_data import load_df
from preprocessing import to_arr, mms_ft, pca_ft, label_ft
from image_processing import generate_bag, generate_histo
from model import run_model

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import cross_val_score


# Load data
data = pd.read_csv(DATA_DF)

#%%
data.columns

# %% Build model

cols = ['LL', 'LH', 'HL', 'HH', 'lbp', 'fft']
features = ['sift', 'surf', 'orb']

all_cols = cols.copy()
all_cols.extend(features)


# try with or without mms, pca
run_model(data, all_cols)



#%%

# Bag-of-words model with SIFT descriptors

sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create(nfeatures=1500)

sift_bag = generate_bag(data['sift'])
surf_bag = generate_bag(data['surf'])
orb_bag = generate_bag(data['orb'])

sift_histo = generate_histo(sift_bag, sift)
surf_histo = generate_histo(surf_bag, surf)
orb_histo = generate_histo(orb_bag, orb)

histo_lists = [sift_histo, surf_histo, orb_histo]

grid2 = []
for histo_list in histo_lists:
    X = np.array(histo_list)
    histo_list = {}

    for name, model in models:
        cv_results = cross_val_score(model, X, y, cv=10, scoring=scoring)
        histo_list[name] = cv_results.mean()

    grid2.append(histo_list)

grid.extend(grid2)

compare = pd.DataFrame()
compare['feature'] = all_cols
compare = compare.set_index('feature')
compare = compare[cols]

# Rearranging
# cols = list(compare.columns)
# cols = [cols[-1]] + cols[:-1]


compare.to_csv('compare_pca_auc.csv')
compare = pd.read_csv('compare_pca.csv')

compare.plot.bar(figsize=(12, 4))
