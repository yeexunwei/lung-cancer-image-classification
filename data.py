#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:50:34 2019

@author: hsunwei
"""
import pandas as pd
from config import DATA_LABEL, DATA_DF, OUTPUT_FOLDER, DTYPE
from import_data import load_scan_df
from image_processing import generate_wavelet, generate_features, generate_lbp, generate_fft

#%%

# Load label data
data = pd.read_csv(DATA_LABEL, dtype=DTYPE)

# Get pixel
data['pixel'] = data.apply(load_scan_df, axis=1)

#%%

# Load features
data = data.apply(generate_wavelet, axis=1)
# data = data.apply(generate_features, axis=1)
# data['lbp'] = data['pixel'].apply(generate_lbp)
# data['fft'] = data['pixel'].apply(generate_fft)

# Save data

# data.to_hdf(OUTPUT_FOLDER + DATA_DF, key='data', mode='w')
data.to_pickle(OUTPUT_FOLDER + 'wavelet.pickle')

# SPIEE =====================================================================
#
# # Load data
#
# df_train = load_df('../CalibrationSet_NoduleData.xlsx')
# df_test = load_df('../TestSet_NoduleData_PublicRelease_wTruth.xlsx')
# data = concat_df(df_train, df_test)
#%%

