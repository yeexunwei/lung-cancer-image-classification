#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 00:05:18 2019

@author: hsunwei
"""
import os
import numpy as np

FOLDER = '../lidc_python/'
INPUT_FOLDER = '../data/LIDC-IDRI/'
OUTPUT_FOLDER = 'lidc_data/'
IMAGE_OUTPUT = '/image_output/'


# label
DATA_LABEL = FOLDER + 'malignancy.csv'
DTYPE = {
    'case': np.int16,
    'roi': np.int16,
    'scan': np.int32,
    'volume': np.float16,
    'diameter': np.float16,
    'x': np.int16,
    'y': np.int16,
    'z': np.int16,
    'malignancy_prob': np.float16,
    'malignancy': np.float16
}
Y_LABEL = OUTPUT_FOLDER + 'malignancy.pkl'

# image pixel
DATA_DF = OUTPUT_FOLDER + "data.pkl"

# image transformation
WAVELET_DF = OUTPUT_FOLDER + "wavelet.pkl"
WAVELET2_DF = OUTPUT_FOLDER + "wavelet2.pkl"
WAVELET3_DF = OUTPUT_FOLDER + "wavelet3.pkl"



SCORING = ["accuracy", "f1", "precision", "recall", "roc_auc"]
# SCORING = ["accuracy_score", "confusion_matrix", "f1_score", "recall_score", "roc_auc_score"]

patients = os.listdir(INPUT_FOLDER)
patients.sort()

# SPIEE =====================================================================
#
# INPUT_FOLDER = '../data/SPIE-AAPM/SPIE-AAPM Lung CT Challenge/'
#
# DATA_DF = "data_df.pickle"
# IMAGE_OUTPUT = '/image_output/'
# SCORING = ["accuracy", "f1", "precision", "recall", "roc_auc"]
# # SCORING = ["accuracy_score", "confusion_matrix", "f1_score", "recall_score", "roc_auc_score"]
#
# patients = os.listdir(INPUT_FOLDER)
# patients.sort()
