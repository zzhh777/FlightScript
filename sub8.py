import pandas as pd
import numpy as np
from sklearn import preprocessing
#from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn import cross_validation
from sklearn import linear_model

def group_WIFI(x):
    x[0] = x[0].replace('<', '-')
    s = x[0].split('-')
    x[0] = s[0] + s[1]
    return x

#init predictor
clus_num = 11
max_abs_scaler = preprocessing.MaxAbsScaler()
enc = preprocessing.OneHotEncoder(sparse = False)
#clus = GaussianMixture(n_components = clus_num)
#clus = KMeans(n_clusters = clus_num)
clus = AffinityPropagation(damping=0.99)

#init data
wifi = pd.read_csv('data/wifi_AP_2D.csv')
wifi_mean = pd.read_csv('data/mean_in_day.csv')

tag = wifi[['WIFIAPTag']].apply(group_WIFI, axis = 1)
tag1 = tag.drop_duplicates()
dic_tag = dict(zip(tag1['WIFIAPTag'], range(0, tag1.__len__())))
tag = tag.replace(dic_tag)
tag = enc.fit_transform(tag.as_matrix())

##############
#clusting data
#############
#normal cluster
# #clust_X = np.append(tag, wifi.drop(['WIFIAPTag'], axis = 1).ix[:, 0 : ], axis = 1)
# clust_X = wifi.drop(['WIFIAPTag'], axis = 1).ix[:, 0 : ]
# clust_X = max_abs_scaler.fit_transform(clust_X)
# clus.fit(clust_X)
# wifi['group'] = clus.predict(clust_X)

#AP cluster
clust_X = wifi.drop(['WIFIAPTag'], axis = 1).T.corr()
wifi['group'] = clus.fit_predict(clust_X)
clus_num = wifi['group'].max()

#mix Gaussian
#clust_X = np.append(tag, wifi.drop(['WIFIAPTag'], axis = 1).ix[:, 0 : ], axis = 1)
#clust_X = wifi.drop(['WIFIAPTag'], axis = 1).ix[:, 0 : ]
#clust_X = max_abs_scaler.fit_transform(clust_X)
#wifi['group'] = clus.predict_proba(clust_X)


#get train data
wifi_t = wifi.drop(['WIFIAPTag'], axis = 1)
for i in [x for x in range(144, 553) if (x % 10 == 0) & (x != 0)]:
    #print(i)
    if i not in [540, 550]:
    #if i not in [200, 320, 340, 360, 480, 500, 540, 550]:
        if i != 150:
            X1 = wifi_t.ix[:, i - 3 : i + 18]
            X1 = np.append(wifi_mean.ix[:, i - 144 - 2 : i - 144 + 20], X1, axis = 1)
            #X1 = np.append(np.tile(wifi_t.ix[:, i - 1].T, (749, 1)), X1, axis = 1)
            X1 = np.append(wifi_t[['group']].as_matrix(), X1, axis = 1)
            X = np.append(X, X1, axis = 0)
        else:
            X = wifi_t.ix[:, i - 3 : i + 18]
            X = np.append(wifi_mean.ix[:, i - 144 - 2 : i - 144 + 20], X, axis = 1)
            #X = np.append(np.tile(wifi_t.ix[:, i - 1].T, (749, 1)), X, axis = 1)
            X = np.append(wifi_t[['group']].as_matrix(), X, axis = 1)

f_score = np.zeros(clus_num + 1)

i = 553
X_test = wifi_t.ix[:, i - 3 : i]
X_test = np.append(wifi_mean.ix[:, i - 144 - 2 : i - 144 + 20], X_test, axis = 1)
#X_test = np.append(np.tile(wifi_t.ix[:, i - 1].T, (749, 1)), X_test, axis = 1)
X_test = np.append(wifi_t.ix[:, 'group'].reshape([749, 1]), X_test, axis = 1)
X_test = np.append(np.array(range(0, 749)).reshape([749, 1]), X_test, axis = 1)

for k in range(0, clus_num + 1):
    #train k th predictor:
    X_t = X[X[:, 0] == k]
    train_X = X_t[:,1 : -18]
    train_Y = X_t[:, -18 : ]
    print(train_X.shape)
    print(train_Y.shape)
    clf = linear_model.MultiTaskElasticNet(max_iter=10000)
    clf.fit(train_X, train_Y)
    # score = cross_validation.cross_val_score(clf, train_X, train_Y, cv = 10, scoring = 'mean_squared_error')
    # print("time: " + str(i) + " Multi_Elastic:" + str(score.mean()))
    # f_score[k] = score.mean() * train_X.shape[0]

    #predict:
    X_t = X_test[X_test[:, 1] == k]
    index = X_t[:, 0]
    if k != 0:
        res_t = clf.predict(X_t[:, 2:])
        res_t = np.append(index.reshape(X_t.shape[0], 1), res_t, axis = 1)
        res = np.append(res_t, res, axis = 0)
    else:
        res = clf.predict(X_t[:, 2:])
        res = np.append(index.reshape(X_t.shape[0], 1), res, axis = 1)

print("all score:" + str(f_score.sum() / X.shape[0]))
res = pd.DataFrame(res)
res = res.sort(0, axis = 0)
res = res.drop([0], axis = 1).as_matrix()
res[res < 0] = 0

sub = pd.read_csv('data/sub_template.csv')
res = res.reshape(1, res.shape[0] * res.shape[1])
sub['passengerCount'] = res[0]
sub.to_csv('sub8_Mix_MultiElastic.csv', index = False)