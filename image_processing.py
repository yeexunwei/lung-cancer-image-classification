"""
Created on Sun Oct  6 22:12:23 2019

@author: hsunwei
"""
import numpy as np
import cv2
import pywt
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from config import IMAGE_OUTPUT


"""Pixel class
"""


class Pixel:
    def __init__(self, img):
        self.img = img
        self.img_int = np.uint8(self.img)

    def plot(self):
        plt_img(self.img, save_as='original')

    def wavelet(self):
        LL, (LH, HL, HH) = wavelet_trans(self.img)
        titles = [' Horizontal detail', 'Vertical detail', 'Diagonal detail']
        plt_img(LH, HL, HH, save_as='wavelet', titles=titles)

    def fft(self):
        magnitude_spectrum = fft_trans(self.img)
        plt_img(magnitude_spectrum, save_as='fft', titles=['Magnitude Spectrum'])

    def lbp(self):
        texture = lbp_ext(self.img)
        plt_img(texture, save_as='lbp', titles=['Texture'])

    def descriptors(self):
        keypoints_sift, descriptors_sift = sift_ext(self.img)
        keypoints_surf, descriptors_surf = surf_ext(self.img)
        keypoints_orb, descriptors_orb = orb_ext(self.img)

        img_int = self.img_int

        plt_img(cv2.drawKeypoints(img_int, keypoints_sift, None, color=(0, 255, 0), flags=0),
                cv2.drawKeypoints(img_int, keypoints_surf, None, color=(0, 255, 0), flags=0),
                cv2.drawKeypoints(img_int, keypoints_orb, None, color=(0, 255, 0), flags=0),
                titles=['SIFT', 'SURF', "ORB"],
                save_as='local_descriptors'
                )


# Plot, save image
def plt_img(*img, save_as=None, titles=None, gray=True, large=True):
    # config
    plt.figure(figsize=(3, 3)) if len(img) == 1 else plt.figure(figsize=(10, 3))
    if large:
        plt.figure(figsize=(8, 4))
    color_map = plt.cm.gray if gray else plt.cm.viridis

    # subplots, plot
    total = len(img)
    for i in range(total):
        num = int('1' + str(total) + str(i + 1))
        ax = plt.subplot(num)
        if titles:
            ax.set_title(titles[i])
        ax.imshow(img[i], cmap=color_map)
    plt.tight_layout()

    # save
    if save_as:
        plt.savefig(IMAGE_OUTPUT + save_as, format='svg', dpi=1200)
    plt.show()
    return


"""Image transformation and extraction
"""


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


def sift_des(original):
    keypoints, descriptors = sift_ext(original)
    return descriptors


def surf_des(original):
    keypoints, descriptors = surf_ext(original)
    return descriptors


def orb_des(original):
    keypoints, descriptors = orb_ext(original)
    return descriptors


"""Generate df
"""


def generate_wavelet(df):
    LL, (LH, HL, HH) = wavelet_trans(df['pixel'])
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
    # initialize feature extraction method
    if ext == 'sift':
        ext = sift_ext
        desc = sift_des

    elif ext == 'surf':
        ext = surf_ext
        desc = surf_des

    elif ext == 'orb':
        ext = orb_ext
        desc = orb_des
    else:
        return

    # convert to descriptors
    img_des = img_col.apply(desc)

    # generate bag of words
    bag = generate_bag(img_des)


    # clustering
    k = 2 * 10
    kmeans = KMeans(n_clusters=k).fit(bag)
    kmeans.verbose = False

    # generate histogram/feature vector
    histo_list = []
    for img in img_col:
        img = np.uint8(img)
        kp, des = ext(img)

        histo = np.zeros(k)
        nkp = np.size(kp)

        for d in des:
            idx = kmeans.predict([d])
            histo[idx] += 1 / nkp  # Because we need normalized histograms, I prefere to add 1/nkp directly

        histo_list.append(histo)
    return histo_list
