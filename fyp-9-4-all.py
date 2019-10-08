from preprocessing import plt_img

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#pca = PCA(125)
# pca = PCA(n_components=125, svd_solver='randomized', whiten=True)



# In[4]:

# In[14]:





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





