"""
Created on Mon Oct  7 15:50:34 2019

@author: hsunwei
"""
import numpy as np
import pandas as pd
import os
from config import DATA_CSV, PIXEL_ARRAY, DTYPE, Y_LABEL
from config import FEATURES, OUTPUT_FOLDER, FORMAT
from import_data import load_scan_df
from image_processing import generate_lbp, generate_fft
from image_processing import generate_ll, generate_lh, generate_hl, generate_hh
from image_processing import generate_sift, generate_surf, generate_orb

PKL = [OUTPUT_FOLDER + feature + FORMAT for feature in FEATURES]
PKL2 = [OUTPUT_FOLDER + feature + '2' + FORMAT for feature in FEATURES]
PKL3 = [OUTPUT_FOLDER + feature + '3' + FORMAT for feature in FEATURES]


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
        # print(df.info())
        np.save(filename, df.iloc[:, 0])
        # df.iloc[:,0].to_pickle(filename)
        del df
    return


if __name__ == '__main__':
    """Get pixel
    """
    try:
        pixel = np.load(PIXEL_ARRAY, allow_pickle=True)
        print(PIXEL_ARRAY + " success read")
    except:
        print(PIXEL_ARRAY + " does not exist, generating a new one...")
        # load label data
        data = pd.read_csv(DATA_CSV, dtype=DTYPE)

        # load pixel according to slice number
        pixel = data.apply(load_scan_df, axis=1)
        y = data['malignancy']
        np.save(PIXEL_ARRAY, pixel)
        np.save(Y_LABEL, y)
        # pixel.to_pickle(PIXEL_ARRAY)
        # y.to_pickle(Y_LABEL)

    """Generate files
    """
    for item in PKL:
        generate_series(item, pixel)

    del pixel
    ll1 = OUTPUT_FOLDER + 'll' + FORMAT
    ll1 = np.load(ll1, allow_pickle=True)
    for filename in PKL2:
        generate_series(filename, ll1)

    del ll1
    ll2 = OUTPUT_FOLDER + 'll2' + FORMAT
    ll2 = np.load(ll2, allow_pickle=True)
    for filename in PKL3:
        generate_series(filename, ll2)
    del ll2

    """Generate histograms
    """
    for root, dirs, files in os.walk(OUTPUT_FOLDER):
        for file in files:
            if 'sift' in file and file.endswith(FORMAT):
                print(file[:-4])
