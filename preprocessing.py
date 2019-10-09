#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 00:05:06 2019

@author: hsunwei
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder


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
    return

#     
    
def to_arr(col):
    arr = np.stack(i.flatten() for i in col)
    return arr

def sc_ft(arr):
    sc = StandardScaler() 
    arr = sc.fit_transform(arr)
    return arr

def mms_ft(arr):
    mms = MinMaxScaler()
    arr = mms.fit_transform(arr)
    return arr

def label_ft(y):
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(y)
    return y

def lda_ft(arr, y):
    lda = LinearDiscriminantAnalysis(n_components=2)
    arr = lda.fit_transform(arr, y)
    return arr, y

def pca_curve(data):
    pca = PCA().fit(data)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    return

def pca_ft(arr):
#     arr = np.stack(i.flatten() for i in col)
    
    # finding the best dimensions
    pca_dims = PCA()
    pca_dims.fit(arr)
    cumsum = np.cumsum(pca_dims.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1
    
    pca_curve(arr)
#     print("number of components: {}".format(d))
    
    # transformation
    pca = PCA(n_components=d)
    components = pca.fit_transform(arr)
    projected = pca.inverse_transform(components)
    
    print("reduced shape: " + str(components.shape))
    print("recovered shape: " + str(projected.shape))

#     plt_img(arr[0].reshape((512,512)), projected[0].reshape((512,512)),
#             save_as='pca', titles=['original', 'pca compressed'])
    return components