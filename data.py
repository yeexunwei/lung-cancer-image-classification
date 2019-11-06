#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:50:34 2019

@author: hsunwei
"""
import pandas as pd
import os
from config import DATA_LABEL, DATA_DF, DTYPE, Y_LABEL
from config import WAVELET_DF, WAVELET2_DF, WAVELET3_DF
from import_data import load_scan_df
from image_processing import generate_wavelet, generate_features, generate_lbp, generate_fft

# %%

# Load label data
data = pd.read_csv(DATA_LABEL, dtype=DTYPE)
data = data[(data['malignancy'] == 0) | (data['malignancy'] == 1)]

# %%

# Get pixel
try:
    pixel = pd.read_pickle(DATA_DF)
except:
    print(DATA_DF + " does not exists, generating a new one...")
    pixel = data.apply(load_scan_df, axis=1)
    y = data['malignancy']
    pixel.to_pickle(DATA_DF)
    y.to_pickle(Y_LABEL)


# %%

# Get wavelets
def wavelet_df(filename, pixel):
    try:
        wavelet = pd.read_pickle(filename)
    except:
        print(filename + " does not exists, generating a new one...")
        wavelet = pd.DataFrame({'pixel': pixel})
        wavelet = wavelet.apply(generate_wavelet, axis=1)
        wavelet.drop(['pixel'], inplace=True, axis=1)
        wavelet.to_pickle(filename)
    return wavelet


wavelet = wavelet_df(WAVELET_DF, pixel)

wavelet2 = wavelet_df(WAVELET2_DF, wavelet['LL'])
del wavelet

wavelet3 = wavelet_df(WAVELET3_DF, wavelet2['LL'])
del wavelet2
del wavelet3

# if not os.path.isfile(WAVELET_DF):
#     print(WAVELET_DF + " does not exists, generating a new one...")
#     wavelet = pd.DataFrame({'pixel':pixel})
#     wavelet = wavelet.apply(generate_wavelet, axis=1)
#     wavelet.drop(columns=['pixel'], inplace=True)
#     wavelet.to_pickle(WAVELET_DF)
#     del wavelet


# %%

#
# # Load label data
# data = pd.read_csv(DATA_LABEL, dtype=DTYPE)
#
# # Get pixel
# data['pixel'] = data.apply(load_scan_df, axis=1)

# %%

# # Load features
# data = data.apply(generate_wavelet, axis=1)
# # data = data.apply(generate_features, axis=1)
# # data['lbp'] = data['pixel'].apply(generate_lbp)
# # data['fft'] = data['pixel'].apply(generate_fft)
#
# # Save data
#
# # data.to_hdf(OUTPUT_FOLDER + DATA_DF, key='data', mode='w')
# data.to_pickle(OUTPUT_FOLDER + 'wavelet.pickle')

# SPIEE =====================================================================
#
# # Load data
#
# df_train = load_df('../CalibrationSet_NoduleData.xlsx')
# df_test = load_df('../TestSet_NoduleData_PublicRelease_wTruth.xlsx')
# data = concat_df(df_train, df_test)
# %%
