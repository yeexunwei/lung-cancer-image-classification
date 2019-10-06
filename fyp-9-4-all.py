from import_data import load_scan, get_pixels_hu


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import exposure





pca = PCA(125)
# pca = PCA(n_components=125, svd_solver='randomized', whiten=True)

#%%

# Example loading of patient 1
first_patient = load_scan(patients[0])
first_patient_pixels = get_pixels_hu(first_patient)

# Basic info of total scans
print('Total number of patients: {}'.format(len(patients)))
print('First patient: {}'.format(patients[0]))
print('Total number of slices: {}'.format(len(first_patient)))



# Sample
original = load_scan_num(81, patients[0])
original2 = load_scan_num(82, patients[0])

plt_img(*data['pixel'][0:3])


# Normalizes the pixel values of the image
plt_img(exposure.equalize_adapthist(data['pixel'][0]))


data = data.apply(generate_wavelet, axis=1)
data = data.apply(generate_sift, axis=1)
data['lbp'] = data['pixel'].apply(generate_lbp)
data['fft'] = data['pixel'].apply(generate_fft)

data.head(3)


# ### save data

# In[14]:


data.to_pickle("signal_image")


# In[3]:


data = pd.read_pickle("signal_image")


#%%

# Wavelet

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
LL, LH, HL, HH = wavelet_trans(original)

plt_img(*[LL, LH, HL, HH], save_as='wavelet', titles=titles)




#%%

# SIFT, SURF, ORB

# transformation of first image
keypoints_sift, descriptors_sift = sift_trans(original)
keypoints_surf, descriptors_surf = surf_trans(original)
keypoints_orb, descriptors_orb = orb_trans(original)

# transformation of second image for comparison
keypoints_sift2, descriptors_sift2 = sift_trans(original2)

plt_img(original,
        cv2.drawKeypoints(original, keypoints_sift, None),
        cv2.drawKeypoints(original, keypoints_surf, None),
        cv2.drawKeypoints(original, keypoints_orb, None),
        titles = ['Original', 'SIFT', 'SURF', "ORB"],
        save_as = 'local_descriptors'
       )

matcher(original, original2, n_matches=80, transformer='surf')


# In[123]:


print('Length of a descriptor: {}'.format(len(descriptors_sift)))
print('Shape of a descriptor: {}'.format(descriptors_sift[0].shape))

plt_img(descriptors_sift[0].reshape(16,8))


# ### LBP



# In[15]:


# settings for LBP
radius = 3
n_points = 8 * radius
METHOD = 'uniform'

lbp = local_binary_pattern(original, n_points, radius, METHOD)


image = original
# image = rgb2gray(image)
# lbp = local_binary_pattern(image, n_points, radius, METHOD)

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


# ### FFT

# numpy fft transformation
f = np.fft.fft2(original)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt_img(original, magnitude_spectrum, save_as='fft', titles=['Original', 'Magnitude Spectrum'])


# Reconstruct the image

# ## Preprocessesing

# In[4]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score, precision_score, roc_curve, roc_auc_score


# In[5]:


def to_arr(col):
    arr = np.stack(i.flatten() for i in col)
    return arr

def sc_trans(arr):
    sc = StandardScaler() 
    arr = sc.fit_transform(arr)
    return arr

def pca_curve(data):
    pca = PCA().fit(data)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    return

def pca_trans(arr):
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


# ### Encode categorical class labels

# In[13]:


X = data['fft']
X = to_arr(X)

y = data['diagnosis']
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# In[15]:


# test model
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)


# In[16]:


probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()


# ### Standard Scaler

# In[29]:


sc = StandardScaler() 
msc = MinMaxScaler()

omin = original.min()
omax = original.max()
xformed = (original - omin)/(omax - omin)


# In[11]:


plt_img(original, xformed, sc_trans(original), msc.fit_transform(original))


# ### PCA for dimensionality reduction

# In[11]:


pca_trans(data.pixel)


# ### LDA

# In[20]:


lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train, y)


# In[19]:


lda.fit(X, y)


# ## Model

# In[58]:


# create all the machine learning models
models = []
models.append(('NB', GaussianNB()))
models.append(('LSVC', LinearSVC()))
models.append(('SVC', SVC(random_state=9, gamma='scale')))
# models.append(('LiR', LinearRegression()))
models.append(('LoR', LogisticRegression(random_state=9, solver='lbfgs')))
models.append(('SGD', SGDClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=9)))
models.append(('RF', RandomForestClassifier(n_estimators=100, random_state=9)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('MLP', MLPClassifier()))

# variables to hold the results and names
results = []
names = []
scoring = "roc_auc"


# In[59]:


features = ['LL','LH','HL','HH','lbp','fft']
features2 = ['sift','surf','orb']

all_features = features.copy()
all_features.extend(features2)

grid = []

for feature in features:
    print(feature)
    X = data[feature]
    X = to_arr(X)
    X = sc_trans(X)
    X = pca_trans(X)

    feature = {}
    
    for name, model in models:
        cv_results = cross_val_score(model, X, y, cv=10, scoring=scoring)
        feature[name] = cv_results.mean()
        
    grid.append(feature)


# #### Bag-of-words model with SIFT descriptors

# In[28]:


def generate_bag(data):
    bag = []
    for row in data:
        for i in row:
            bag.append(i)
    return bag

def generate_histo(bag, local_desc):
    k = 2 * 10
    kmeans = KMeans(n_clusters=k).fit(bag)
    kmeans.verbose = False
#     sift = cv2.xfeatures2d.SIFT_create()

    histo_list = []

    for img in data['pixel']:
        img = np.uint8(img)
        kp, des = local_desc.detectAndCompute(img, None)

        histo = np.zeros(k)
        nkp = np.size(kp)

        for d in des:
            idx = kmeans.predict([d])
            histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly

        histo_list.append(histo)
    return histo_list


# In[29]:


sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create(nfeatures=1500)

sift_bag = generate_bag(data['sift'])
surf_bag = generate_bag(data['surf'])
orb_bag = generate_bag(data['orb'])


# In[30]:


sift_histo = generate_histo(sift_bag, sift)
surf_histo = generate_histo(surf_bag, surf)
orb_histo = generate_histo(orb_bag, orb)

histo_lists= [sift_histo, surf_histo, orb_histo]


# In[60]:


grid2 = []
for histo_list in histo_lists:
    X = np.array(histo_list)
    histo_list = {}
    
    for name, model in models:
        cv_results = cross_val_score(model, X, y, cv=10, scoring=scoring)
        histo_list[name] = cv_results.mean()
        
    grid2.append(histo_list)


# In[62]:


grid.extend(grid2)

compare['feature'] = all_features
cols = list(compare.columns)
cols = [cols[-1]] + cols[:-1]
compare = compare[cols]

compare = compare.set_index('feature')


# In[63]:


compare


# In[64]:


compare.to_csv('compare_pca_auc.csv')



compare = pd.read_csv('compare_pca.csv')


# In[65]:


compare.plot.bar(figsize=(12,4))





