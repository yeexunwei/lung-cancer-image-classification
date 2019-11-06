#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 23:27:49 2019

@author: hsunwei
"""

# https://github.com/WillKoehrsen/machine-learning-project-walkthrough/blob/master/Machine%20Learning%20Project%20Part%201.ipynb
# https://github.com/WillKoehrsen/machine-learning-project-walkthrough

# %%
from importlib import reload
import model
reload(model)

from config import DATA_DF, Y_LABEL
from import_data import read_df
from preprocessing import label_ft
from image_processing import generate_histo
from model import run_model, run_desc_model
import pandas as pd

# Load data
y = pd.read_pickle(Y_LABEL)
y = label_ft(y)


#%%

# Build model

cols = ['LL', 'LH', 'HL', 'HH', 'lbp', 'fft']
features = ['sift', 'surf', 'orb']

all_cols = cols.copy()
all_cols.extend(features)

# %%
# build models of attributes

import model
reload(model)
result_list1 = model.run_model(data[cols], y)

#%%
print(result_list1)



# %% Bag-of-words model with feature descriptors

sift_histo = generate_histo(data['pixel'], "sift")
surf_histo = generate_histo(data['pixel'], "surf")
orb_histo = generate_histo(data['pixel'], "orb")

histo_lists = [sift_histo, surf_histo, orb_histo]

# build models
result_list2 = run_desc_model(histo_lists, y)

# %%
all_results = result_list1.copy()
all_results.extend(result_list2)

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