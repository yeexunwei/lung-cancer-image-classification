"""
Created on Mon Oct  7 00:05:18 2019

@author: hsunwei
"""
import os
import numpy as np

FOLDER = '../lidc_preprocessing/'
INPUT_FOLDER = '../data/LIDC-IDRI/'
OUTPUT_FOLDER = 'lidc_data/'
IMAGE_OUTPUT = '/image_output/'

FORMAT = ".npy"
TRANSFORMATION_LIST = ['LL', 'LH', 'HL', 'HH', 'lbp', 'fft']
EXTRACTION_LIST = ['sift', 'surf', 'orb']

SCORING = ["accuracy", "f1", "precision", "recall", "roc_auc"]

# label
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
# image pixel
PIXEL_ARRAY = OUTPUT_FOLDER + "pixel" + FORMAT
Y_LABEL = OUTPUT_FOLDER + 'malignancy' + FORMAT

# image transformation
FEATURES = ['ll', 'lh', 'hl', 'hh', "fft", "lbp", "surf", "sift", "orb"]

patients = os.listdir(INPUT_FOLDER)
patients.sort()
