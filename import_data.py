"""
Created on Fri Oct  4 23:04:35 2019

@author: hsunwei
"""
from config import INPUT_FOLDER, patients

import glob
import numpy as np
import pandas as pd
import pydicom


def scan_to_df(df):
    # Load pixels for dataframe
    return get_scan_num(df['case'], df['slice_no'])


def get_scan_num(case, slice_no):
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