"""
Created on Mon Oct  7 00:05:06 2019

@author: hsunwei
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA


# Unflatten Series
def to_arr(col):
    arr = np.stack(i.flatten() for i in col)
    return arr


def pca_curve(data):
    pca = PCA().fit(data)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    return


def pca_ft(arr):
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
