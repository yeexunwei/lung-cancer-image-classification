#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:50:34 2019

@author: hsunwei
"""
from config import DATA_PICKLE
from import_data import load_df, concat_df
#from image_processing import generate_wavelet, generate_sift, generate_lbp, generate_fft


#%% Load data

df_train = load_df('../CalibrationSet_NoduleData.xlsx')
df_test = load_df('../TestSet_NoduleData_PublicRelease_wTruth.xlsx')
data = concat_df(df_train, df_test)
#
#data = data.apply(generate_wavelet, axis=1)
#data = data.apply(generate_sift, axis=1)
#data['lbp'] = data['pixel'].apply(generate_lbp)
#data['fft'] = data['pixel'].apply(generate_fft)

#%% Save data

data.to_pickle(DATA_PICKLE)