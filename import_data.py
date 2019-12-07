#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 23:04:35 2019

@author: hsunwei
"""
from config import INPUT_FOLDER, patients

import glob
import numpy as np
import pandas as pd
import pydicom


def load_scan_df(df):
    # Load pixels for dataframe
    return load_scan_num(df['case'], df['slice_no'])


def load_scan_num(case, slice_no):
    # Load the specified slice in given folder path and return pixel array
    path = get_path(case, slice_no)
    dicom = pydicom.dcmread(path)
    image_array = get_pixel_hu(dicom)
    return image_array


def get_path(case, slice_no):
    # Generate path according to case and slice no
    case = str(case)
    slice_no = str(slice_no)
    case = case.zfill(4)
    slice_no = slice_no.zfill(4)

    return glob.glob(INPUT_FOLDER + "*" + case + "/*/*/*" + slice_no + ".dcm")[0]


def get_pixel_hu(dicom):
    # Pixel to np.array

    image = dicom.pixel_array
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = dicom.RescaleIntercept
    slope = dicom.RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)




# SPIEE =====================================================================
#
# # Load the number of slices in given folder path
# def load_scan_count(path):
#     slices = [s for s in glob.glob(INPUT_FOLDER + path + '/*/*/*.dcm')]
#     return len(slices)
#
#
# # Load the specified slice in given folder path
# def load_scan_num(num, path):
#     # print(glob.glob(INPUT_FOLDER + path + '/*/*/*' + str(num) + '.dcm'))
#     slices = pydicom.dcmread(glob.glob(INPUT_FOLDER + path + '/*/*/*' + str(num) + '.dcm')[0])
#     return slices.pixel_array
#
#
# # Load pixels for dataframe
# def load_scan_df(df):
#     return load_scan_num(df['z'], df['path'])
#
#
#
# # Load label data
#
# def load_df(file):
#     path = INPUT_FOLDER + file
#
#     df = pd.read_excel(path, dtype={'Nodule Center x,y Position*': str})
#     df.rename(columns={'Scan Number': 'patient_id',
#                        'Nodule Number': 'nodule_number',
#                        'Diagnosis': 'diagnosis',
#                        'Final Diagnosis': 'diagnosis'},
#               inplace=True)
#
#     df.dropna(subset=['diagnosis'], inplace=True)
#     df['diagnosis'] = np.where(df['diagnosis'].str.contains('benign', False), 0, 1)
#     df['diagnosis'] = df['diagnosis'].astype('category')
#
#     df['x'] = df['Nodule Center x,y Position*'].apply(lambda x: x[:-3])
#     df['x'] = df['x'].str.replace(",", "")
#     df['y'] = df['Nodule Center x,y Position*'].apply(lambda x: x[-3:])
#     df['z'] = df['Nodule Center Image'].astype('int')
#     df.drop(columns=['Nodule Center x,y Position*'], inplace=True)
#     df.drop(columns=['Nodule Center Image'], inplace=True)
#
#     if 'nodule_number' in df.columns:
#         df['nodule_number'] = df['nodule_number'].astype('int')
#     df.patient_id = df.patient_id.str.lower()
#     return df
#
#
# def concat_df(df_train, df_test):
#     # Append df_train and df_test
#     table = df_train.append(df_test)
#     table['nodule_number'].fillna('1', inplace=True)
#     table = table[['patient_id', 'nodule_number', 'diagnosis', 'x', 'y', 'z']]
#
#     # Convert patients to df and append data
#     data = pd.DataFrame(patients, columns=['path'])
#     data['patient_id'] = data['path'].str.lower()
#     data = pd.merge(data, table, on='patient_id')
#
#     # Load pixel_array for each row
#     data['pixel'] = data.apply(load_scan_df, axis=1)
#     # data['pixel_flatten'] = data.pixel.values.reshape(-1)
#     # data['pixel_flatten'] = data.pixel.apply(np.ndarray.flatten) #.apply(np.ndarray.tolist)
#     # data['pixel_flatten'] = data.pixel_flatten.apply(np.ndarray.tolist)
#     return data

# %%
