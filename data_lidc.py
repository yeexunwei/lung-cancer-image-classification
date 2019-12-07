#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:50:34 2019

@author: hsunwei
"""
import numpy as np
import pandas as pd
import os
from config import DATA_LABEL, DATA_DF, DTYPE, Y_LABEL
from config import LIST, OUTPUT_FOLDER, FORMAT
from import_data import load_scan_df
from image_processing import generate_wavelet, generate_features, generate_lbp, generate_fft
from image_processing import generate_ll, generate_lh, generate_hl, generate_hh
from image_processing import generate_sift, generate_surf, generate_orb

PKL = [OUTPUT_FOLDER + item + FORMAT for item in LIST]
PKL2 = [OUTPUT_FOLDER + item + FORMAT for item in LIST]
PKL3 = [OUTPUT_FOLDER + item + FORMAT for item in LIST]

# Get wavelets
def generate_series(filename, pixel):
    if "ll" in filename:
        method = generate_ll
    elif "lh" in filename:
        method = generate_lh
    elif "hl" in filename:
        method = generate_hl
    elif "hh" in filename:
        method = generate_hh

    elif "fft" in filename:
        method = generate_fft
    elif "lbp" in filename:
        method = generate_lbp
    elif "sift" in filename:
        method = generate_sift
    elif "surf" in filename:
        method = generate_surf
    elif "orb" in filename:
        method = generate_orb

    if os.path.isfile(filename) and os.path.getsize(filename) > 0:
        print(filename + " exists")
    else:
        print(filename + " does not exist, generating a new one...")
        df = pd.DataFrame({'pixel': pixel})
        df = df.apply(method, axis=1)
        df.drop(['pixel'], inplace=True, axis=1)
        print(df.info())
        np.save(filename, df.iloc[:,0])
        # df.iloc[:,0].to_pickle(filename)
        del df

    # try:
    #     df = pd.read_pickle(filename)
    #     print(filename + " exists")
    # except:
    #     print(filename + " does not exist, generating a new one...")
    #     df = pd.DataFrame({'pixel': pixel})
    #     df = df.apply(method, axis=1)
    #     df.drop(['pixel'], inplace=True, axis=1)
    #     df.to_pickle(filename)

    return


if __name__ == '__main__':
    # Load label data
    # data = pd.read_csv(DATA_LABEL, dtype=DTYPE)

    # Get pixel
    try:
        pixel = pd.read_pickle(DATA_DF)
        print(DATA_DF + " success read")
    except:
        print(DATA_DF + " does not exist, generating a new one...")
        # load pixel according to slice number
        pixel = data.apply(load_scan_df, axis=1)
        y = data['malignancy']
        pixel.to_pickle(DATA_DF)
        y.to_pickle(Y_LABEL)

    for item in PKL:
        generate_series(item, pixel)

    del pixel
    ll1 = pd.read_pickle(OUTPUT_FOLDER + 'll.pkl')
    for filename in PKL2:
        generate_series(filename, ll1)

    del ll1
    ll2 = pd.read_pickle(OUTPUT_FOLDER + 'll2.pkl')
    for filename in PKL3:
        generate_series(filename, ll2)