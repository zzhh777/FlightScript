import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.ensemble import BaggingRegressor

def group_WIFI(x):
    x[0] = x[0].replace('<', '-')
    s = x[0].split('-')
    x[0] = s[0] + s[1]
    return x

#init predictor
clus_num = 11
max_abs_scaler = preprocessing.MaxAbsScaler()
enc = preprocessing.OneHotEncoder(sparse = False)
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
clus_num = wifi['group'].max() + 1

#get train data
wifi_t = wifi.drop(['WIFIAPTag'], axis = 1)
for i in [x for x in range(150, 553) if (x % 5 == 0) & (x != 0)]:
    #print(i)
    if i not in range(520, 560):
    #if i not in [200, 320, 340, 360, 480, 500, 540, 550]:
        if i != 150:
            X1 = wifi_t.ix[:, i - 1 : i + 18]
            X1 = np.append(wifi_mean.ix[:, i - 144 - 2 : i - 144 + 20], X1, axis = 1)
            #X1 = np.append(np.tile(wifi_t.ix[:, i - 1].T, (749, 1)), X1, axis = 1)
            X1 = np.append(wifi_t[['group']].as_matrix(), X1, axis = 1)
            X = np.append(X, X1, axis = 0)
        else:
            X = wifi_t.ix[:, i - 1 : i + 18]
            X = np.append(wifi_mean.ix[:, i - 144 - 2 : i - 144 + 20], X, axis = 1)
            #X = np.append(np.tile(wifi_t.ix[:, i - 1].T, (749, 1)), X, axis = 1)
            X = np.append(wifi_t[['group']].as_matrix(), X, axis = 1)

f_score = np.zeros((18, clus_num))



i = 521
X_test = wifi_t.ix[:, i - 1 : i]
X_test = np.append(wifi_mean.ix[:, i - 144 - 2 : i - 144 + 20], X_test, axis = 1)
#X_test = np.append(np.tile(wifi_t.ix[:, i - 1].T, (749, 1)), X_test, axis = 1)
X_test = np.append(wifi_t.ix[:, 'group'].reshape([749, 1]), X_test, axis = 1)
X_test = np.append(np.array(range(0, 749)).reshape([749, 1]), X_test, axis = 1)

for k in range(0, clus_num):
    #train k th predictor:
    for j in range(18):
        X_t = X[X[:, 0] == k]
        train_X = X_t[:,1 : -18]
        train_Y = X_t[:, -(18 - j) ]
        #print(train_X.shape)
        #print(train_Y.shape)
        #clf = linear_model.Lasso()
        #clf = linear_model.Ridge(alpha=0.5)
        clf = linear_model.LinearRegression()
        #clf = GradientBoostingRegressor()
        #clf = BaggingRegressor(linear_model.Lasso())
        score = cross_validation.cross_val_score(clf, train_X, train_Y, cv = 10, scoring = 'mean_squared_error')
        #score = clf.score(train_X, train_Y)
        clf.fit(train_X, train_Y)
        print("time: " + str(k) + " Multi_Elastic:" + str(score.mean()))
        f_score[j, k] = score.mean() * train_X.shape[0]

        #predict:
        X_t = X_test[X_test[:, 1] == k]
        if j == 0:
            res_t = np.zeros((X_t.shape[0], 19))
            res_t[:, 0] = X_t[:, 0]
        res_t[:, j + 1] = clf.predict(X_t[:, 2:])
    if k != 0:
        res = np.append(res_t, res, axis = 0)
    else:
        res = res_t
print("all score:" + str(f_score.sum().sum() / X.shape[0] / 18))
res = pd.DataFrame(res)
res = res.sort(0, axis = 0)
res = res.drop([0], axis = 1).as_matrix()
res[res < 0] = 0

#print("test error:" + str(np.power((res - wifi_t.ix[:, 521 : 521 + 18]), 2).sum().sum()))

sub = pd.read_csv('data/sub_template.csv')
res = res.reshape(1, res.shape[0] * res.shape[1])
sub['passengerCount'] = res[0]
sub.to_csv('sub9_Mix_LinearReg.csv', index = False)