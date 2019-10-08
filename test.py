#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:51:38 2019

@author: hsunwei
"""

from config import patients, DATA_PICKLE
from import_data import load_scan, get_pixels_hu, load_scan_num
from image_processing import wavelet_trans, fft_trans
from image_processing import sift_ext, surf_ext, orb_ext, matcher, lbp_ext
from preprocessing import plt_img

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from skimage.color import label2rgb
import cv2

#%%

# Example loading of patient 1
first_patient = load_scan(patients[0])
first_patient_pixels = get_pixels_hu(first_patient)

# Sample
original = load_scan_num(81, patients[0])
original2 = load_scan_num(82, patients[0])

# Basic info of total scans
print('Total number of patients: {}'.format(len(patients)))
print('First patient: {}'.format(patients[0]))
print('Total number of slices: {}'.format(len(first_patient)))

#%%

data = pd.read_pickle(DATA_PICKLE)
plt_img(*data['pixel'][0:3])

# Normalizes the pixel values of the image
plt_img(exposure.equalize_adapthist(data['pixel'][0]))

#%% Wavelet

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail', 'Vertical detail', 'Diagonal detail']
LL, (LH, HL, HH) = wavelet_trans(original)

plt_img(*[LL, LH, HL, HH], save_as='wavelet', titles=titles)

#%% FFT

magnitude_spectrum = fft_trans(original)

plt_img(original, magnitude_spectrum, save_as='fft', titles=['Original', 'Magnitude Spectrum'])

#%% SIFT, SURF, ORB

# transformation of first image
keypoints_sift, descriptors_sift = sift_ext(original)
keypoints_surf, descriptors_surf = surf_ext(original)
keypoints_orb, descriptors_orb = orb_ext(original)

# transformation of second image for comparison
keypoints_sift2, descriptors_sift2 = sift_ext(original2)

plt_img(original,
        cv2.drawKeypoints(original, keypoints_sift, None),
        cv2.drawKeypoints(original, keypoints_surf, None),
        cv2.drawKeypoints(original, keypoints_orb, None),
        titles = ['Original', 'SIFT', 'SURF', "ORB"],
        save_as = 'feature_descriptors'
       )

matcher(original, original2, n_matches=80, transformer='surf')


print('Length of a descriptor: {}'.format(len(descriptors_sift)))
print('Shape of a descriptor: {}'.format(descriptors_sift[0].shape))

plt_img(descriptors_sift[0].reshape(16,8))

#%% LBP

# settings for LBP
radius = 3
n_points = 8 * radius
METHOD = 'uniform'

image = original
lbp = lbp_ext(image)

def overlay_labels(image, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)

def highlight_bars(bars, indexes):
    for i in indexes:
        bars[i].set_facecolor('r')

def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')

# plot histograms of LBP of textures
fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
plt.gray()

titles = ('edge', 'flat', 'corner')
w = width = radius - 1
edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
i_14 = n_points // 4            # 1/4th of the histogram
i_34 = 3 * (n_points // 4)      # 3/4th of the histogram
corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
                 list(range(i_34 - w, i_34 + w + 1)))

label_sets = (edge_labels, flat_labels, corner_labels)

for ax, labels in zip(ax_img, label_sets):
    ax.imshow(overlay_labels(image, lbp, labels))

for ax, labels, name in zip(ax_hist, label_sets, titles):
    counts, _, bars = hist(ax, lbp)
    highlight_bars(bars, labels)
    ax.set_ylim(top=np.max(counts[:-1]))
    ax.set_xlim(right=n_points + 2)
    ax.set_title(name)

ax_hist[0].set_ylabel('Percentage')
for ax in ax_img:
    ax.axis('off')
    
fig.savefig('image_output/lbp', format='svg', dpi=1200)