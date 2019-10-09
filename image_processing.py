#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 22:12:23 2019

@author: hsunwei
"""
from preprocessing import plt_img
import numpy as np

# signal processing
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
def generate_wavelet(df):
    LL, (LH, HL, HH) = wavelet_trans(df['pixel'])
    df['LL'] = LL
    df['LH'] = LH
    df['HL'] = HL
    df['HH'] = HH
    return df


def generate_fft(img):
    return fft_trans(img)


def generate_features(df):
    img = np.uint8(df['pixel'])

    keypoints_sift, descriptors_sift = sift_ext(img)
    keypoints_surf, descriptors_surf = surf_ext(img)
    keypoints_orb, descriptors_orb = orb_ext(img)
    df['sift'] = descriptors_sift
    df['surf'] = descriptors_surf
    df['orb'] = descriptors_orb
    return df


def generate_lbp(img):
    return lbp_ext(img)


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


def generate_bag(data):
    bag = []
    for row in data:
        for i in row:
            bag.append(i)
    return bag


def generate_histo(col, bag, local_desc):
    k = 2 * 10
    kmeans = KMeans(n_clusters=k).fit(bag)
    kmeans.verbose = False
    #     sift = cv2.xfeatures2d.SIFT_create()

    histo_list = []

    for img in col:
        img = np.uint8(img)
        kp, des = local_desc.detectAndCompute(img, None)

        histo = np.zeros(k)
        nkp = np.size(kp)

        for d in des:
            idx = kmeans.predict([d])
            histo[idx] += 1 / nkp  # Because we need normalized histograms, I prefere to add 1/nkp directly

        histo_list.append(histo)
    return histo_list
