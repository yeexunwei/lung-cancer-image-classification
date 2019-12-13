"""
Created on Mon Oct  7 00:05:18 2019

@author: hsunwei
"""
import os
import numpy as np


"Folders"

FOLDER = '../lidc_preprocessing/'
INPUT_FOLDER = '../data/LIDC-IDRI/'
OUTPUT_FOLDER = 'lidc_data/'
IMAGE_OUTPUT = '/image_output/'

FORMAT = ".npy"
SCORING = ["accuracy", "f1", "precision", "recall", "roc_auc"]


"Input"

DATA_CSV = FOLDER + 'malignancy.csv'
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


"Original data"

PIXEL_ARRAY = OUTPUT_FOLDER + "pixel" + FORMAT
Y_LABEL = OUTPUT_FOLDER + 'malignancy' + FORMAT


"All features and filenames"

# TRANSFORMATION_LIST = ['LL', 'LH', 'HL', 'HH', 'lbp', 'fft']
TRANSFORMATION_LIST = ['ll', 'lh', 'hl', 'hh', 'lbp', 'fft']
# TRANSFORMATION_LIST = ['ll', 'lh', 'hl', 'hh']
# TRANSFORMATION_LIST = []
EXTRACTION_LIST = ['sift', 'surf', 'orb']

FEATURES = TRANSFORMATION_LIST.copy()
FEATURES.extend('histo_' + feature for feature in EXTRACTION_LIST)

FEATURES_ARRAY = [OUTPUT_FOLDER + feature + FORMAT for feature in FEATURES]
FEATURES_ARRAY2 = [OUTPUT_FOLDER + feature + '2' + FORMAT for feature in FEATURES]
FEATURES_ARRAY3 = [OUTPUT_FOLDER + feature + '3' + FORMAT for feature in FEATURES]


patients = os.listdir(INPUT_FOLDER)
patients.sort()
