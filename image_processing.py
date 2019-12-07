#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 22:12:23 2019

@author: hsunwei
"""
from preprocessing import plt_img
import numpy as np
import cv2
import pywt
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans


# Image transformation
def wavelet_trans(original):
    coeffs2 = pywt.dwt2(original, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    return LL, (LH, HL, HH)


def fft_trans(original):
    f = np.fft.fft2(original)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum


# Feature extraction
def sift_ext(original):
    sift = cv2.xfeatures2d.SIFT_create()

    img = np.uint8(original)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors


def surf_ext(original):
    surf = cv2.xfeatures2d.SURF_create()

    img = np.uint8(original)
    keypoints, descriptors = surf.detectAndCompute(img, None)
    return keypoints, descriptors


def orb_ext(original):
    orb = cv2.ORB_create(nfeatures=1500)

    img = np.uint8(original)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors


def lbp_ext(original):
    radius = 3
    n_points = 8 * radius
    METHOD = 'uniform'
    lbp = local_binary_pattern(original, n_points, radius, METHOD)
    return lbp


# Generate df
def generate_wavelet(df, type=None):
    LL, (LH, HL, HH) = wavelet_trans(df['pixel'])
    # if type == "ll":
    #     df['LL'] = LL
    # elif type == 'lh':
    #     df['LH'] = LH
    # elif type == 'hl':
    #     df['HL'] = HL
    # elif type == 'hh':
    #     df['HH'] = HH
    # else:
    df['LL'] = LL
    df['LH'] = LH
    df['HL'] = HL
    df['HH'] = HH

    return df

def generate_ll(df):
    LL, (LH, HL, HH) = wavelet_trans(df['pixel'])
    df['LL'] = LL
    return df

def generate_lh(df):
    LL, (LH, HL, HH) = wavelet_trans(df['pixel'])
    df['LH'] = LH
    return df

def generate_hl(df):
    LL, (LH, HL, HH) = wavelet_trans(df['pixel'])
    df['HL'] = HL
    return df

def generate_hh(df):
    LL, (LH, HL, HH) = wavelet_trans(df['pixel'])
    df['HH'] = HH
    return df


def generate_fft(df):
    df['fft'] = fft_trans(df['pixel'])
    return df


def generate_lbp(df):
    df['lbp'] = lbp_ext(df['pixel'])
    return df


def generate_sift(df):
    img = np.uint8(df['pixel'])
    keypoints_sift, descriptors_sift = sift_ext(img)
    df['sift'] = descriptors_sift
    return df


def generate_surf(df):
    img = np.uint8(df['pixel'])
    keypoints_surf, descriptors_surf = surf_ext(img)
    df['surf'] = descriptors_surf
    return df


def generate_orb(df):
    img = np.uint8(df['pixel'])
    keypoints_orb, descriptors_orb = orb_ext(img)
    df['orb'] = descriptors_orb
    return df


def generate_features(df):
    img = np.uint8(df['pixel'])

    keypoints_sift, descriptors_sift = sift_ext(img)
    keypoints_surf, descriptors_surf = surf_ext(img)
    keypoints_orb, descriptors_orb = orb_ext(img)
    df['sift'] = descriptors_sift
    df['surf'] = descriptors_surf
    df['orb'] = descriptors_orb
    return df


# match images using desriptors
def matcher(original, original2, n_matches=80, transformer='sift'):
    img = np.uint8(original)
    img2 = np.uint8(original2)

    if transformer == 'sift':
        trans = sift_ext
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    elif transformer == 'surf':
        trans = surf_ext
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    elif transformer == 'orb':
        trans = orb_ext
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        return None

    keypoints, descriptors = trans(original)
    keypointst2, descriptors2 = trans(original2)

    matches = bf.match(descriptors, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    N_MATCHES = n_matches

    match_img = cv2.drawMatches(
        img, keypoints,
        img2, keypoints,
        matches[:N_MATCHES], img2.copy(), flags=2)

    plt_img(match_img, large=True, save_as=transformer, titles=[transformer + ' match'])

    return


# generate bag-of-words
def generate_bag(data):
    bag = []
    for row in data:
        for i in row:
            bag.append(i)
    return bag


# generate feature vector after clustering
def generate_histo(img_col, ext):
    # generate bag of words
    bag = generate_bag(img_col)

    # initialize feature extraction method
    if ext == 'sift':
        ext = cv2.xfeatures2d.SIFT_create()
    elif ext == 'surf':
        ext = cv2.xfeatures2d.SURF_create()
    elif ext == 'orb':
        ext = cv2.ORB_create(nfeatures=1500)
    else:
        return

    # clustering
    k = 2 * 10
    kmeans = KMeans(n_clusters=k).fit(bag)
    kmeans.verbose = False

    # generate histogram/feature vector
    histo_list = []
    for img in img_col:
        img = np.uint8(img)
        kp, des = ext.detectAndCompute(img, None)

        histo = np.zeros(k)
        nkp = np.size(kp)

        for d in des:
            idx = kmeans.predict([d])
            histo[idx] += 1 / nkp  # Because we need normalized histograms, I prefere to add 1/nkp directly

        histo_list.append(histo)
    return histo_list