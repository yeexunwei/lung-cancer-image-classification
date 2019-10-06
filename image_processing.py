#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 22:12:23 2019

@author: hsunwei
"""

import numpy as np

# signal processing
import cv2
import pywt
from skimage.feature import local_binary_pattern
from skimage.color import label2rgb #rgb2gray


def wavelet_trans(original):
    coeffs2 = pywt.dwt2(original, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    return LL, LH, HL, HH

def generate_wavelet(df):
    coeffs2 = pywt.dwt2(df['pixel'], 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    df['LL'] = LL
    df['LH'] = LH
    df['HL'] = HL
    df['HH'] = HH
    return df

def generate_sift(df):
    sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()
    orb = cv2.ORB_create(nfeatures=1500)

    img = np.uint8(df['pixel'])
    keypoints_sift, descriptors_sift = sift.detectAndCompute(img, None)
    keypoints_surf, descriptors_surf = surf.detectAndCompute(img, None)
    keypoints_orb, descriptors_orb = orb.detectAndCompute(img, None)
    df['sift'] = descriptors_sift
    df['surf'] = descriptors_surf
    df['orb'] = descriptors_orb
    return df

def generate_lbp(image):
    lbp = local_binary_pattern(image, 8* 3, 3, 'uniform')
    return lbp

def generate_fft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    return magnitude_spectrum


# feature extraction
def sift_trans(original):
    sift = cv2.xfeatures2d.SIFT_create()
    
    img = np.uint8(original)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

def surf_trans(original):
    surf = cv2.xfeatures2d.SURF_create()
    
    img = np.uint8(original)
    keypoints, descriptors = surf.detectAndCompute(img, None)
    return keypoints, descriptors

def orb_trans(original):
    orb = cv2.ORB_create(nfeatures=1500)
    
    img = np.uint8(original)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors

def matcher(original, orignal2, n_matches=80, transformer='sift'):
    img = np.uint8(original)
    img2 = np.uint8(original2)
    
    if transformer == 'sift':
        trans = sift_trans
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    elif transformer == 'surf':
        trans = surf_trans
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    elif transformer == 'orb':
        trans = orb_trans
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        return None
        
    keypoints, descriptors = trans(original)
    keypointst2, descriptors2 = trans(original2)
    
    matches = bf.match(descriptors, descriptors2)
    matches = sorted(matches, key = lambda x:x.distance)

    N_MATCHES = n_matches

    match_img = cv2.drawMatches(
        img, keypoints,
        img2, keypoints,
        matches[:N_MATCHES], img2.copy(), flags=2)

    plt_img(match_img, large=True, save_as=transformer, titles=[transformer + ' match'])