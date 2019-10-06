#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 00:05:06 2019

@author: hsunwei
"""

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler


def plt_img(*img, save_as=None, titles=None, gray=True, large=False):
    
    plt.figure(figsize=(3, 3)) if len(img) == 1 else plt.figure(figsize=(10, 3))
    if large == True:
        plt.figure(figsize=(8,4))
    
    color_map = plt.cm.gray if gray else plt.cm.viridis
        
    total = len(img)
    for i in range(total):
        num = int('1' + str(total) + str(i+1))
        ax = plt.subplot(num)
        if titles:
            ax.set_title(titles[i])
        ax.imshow(img[i], cmap=color_map)
        
    plt.tight_layout()
    if save_as:
        plt.savefig('image_output/' + save_as, format='svg', dpi=1200)
    plt.show()