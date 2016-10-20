#思路是将AP分组，然后用线性模型预测
#
#分组的算法尝试了AP聚类和k-means，对于每个AP使用历史中每10分钟的均值作为特征
#AP聚类使用AP特征相关系数矩阵作为相似度
#
#线性模型使用sklearn的MultiElastic
#尝试过直接使用Lasso和Ridge混合，效果会变差一些

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn import cross_validation
from sklearn import linear_model

#init predictor
#初始化聚类器和归一化函数
print('prepareing...')
clus_num = 11
max_abs_scaler = preprocessing.MaxAbsScaler()
enc = preprocessing.OneHotEncoder(sparse = False)
#clus = GaussianMixture(n_components = clus_num)
clus = KMeans(n_clusters = clus_num)
#clus = AffinityPropagation(damping=0.99)

#init data
#读取数据
#连接数据
wifi = pd.read_csv('data/wifi_AP_2D.csv')
wifi_t = wifi.drop(['WIFIAPTag'], axis = 1)
wifi_t.columns = range(2130)
#均值数据
wifi_mean = pd.read_csv('data/mean_in_day.csv')

##############
#clusting data
#############
#k-means聚类
#normal cluster
clust_X = wifi.drop(['WIFIAPTag'], axis = 1).ix[:, 0 : ]
clust_X = max_abs_scaler.fit_transform(clust_X)
clus.fit(clust_X)
wifi['group'] = clus.predict(clust_X)

#AP cluster
# #AP聚类
# clust_X = wifi.drop(['WIFIAPTag'], axis = 1).T.corr()
# wifi['group'] = clus.fit_predict(clust_X)
# wifi_t['group'] = wifi['group']
# clus_num = wifi['group'].max() + 1

#get train data
#采集训练数据，每200分钟采集一次，
print('getting data...')
for i in [x for x in range(150, 2100) if (x % 5 == 0) & (x != 0)]:
    #print(i)
    if i not in range(2100, 2130):
        print(i)
    #if i not in [200, 320, 340, 360, 480, 500, 540, 550]:
        if i != 265:
            #对于第i个时间片，采集i-3 : i-1的连接数量
            X1 = wifi_t.ix[:, i - 3 : i + 18]
            #采集同时段的均值，这里144为一天的时间片数量，由于每144片内的数据都是一样的减不减144都一样
            X1 = np.append(wifi_mean.ix[:, i - 144 - 2 : i - 144 + 20], X1, axis = 1)
            #X1 = np.append(np.tile(wifi_t.ix[:, i - 1].T, (749, 1)), X1, axis = 1)
            X1 = np.append(wifi_t[['group']].as_matrix(), X1, axis = 1)
            X = np.append(X, X1, axis = 0)
        else:
            #对于第一条数据特殊处理
            X = wifi_t.ix[:, i - 3 : i + 18]
            X = np.append(wifi_mean.ix[:, i - 2 : i + 20], X, axis = 1)
            #X = np.append(np.tile(wifi_t.ix[:, i - 1].T, (749, 1)), X, axis = 1)
            X = np.append(wifi_t[['group']].as_matrix(), X, axis = 1)

print(X.shape)
f_score = np.zeros(clus_num)
# X = pd.DataFrame(X)
# X = X.fillna(0)
# X = X.as_matrix()

# #形成测试集/预测目标
# i = 2130 - 144
# X_test = wifi_t.ix[:, i - 3 : i]
# X_test = np.append(wifi_mean.ix[:, i - 2 : i + 20], X_test, axis = 1)
# #X_test = np.append(np.tile(wifi_t.ix[:, i - 1].T, (749, 1)), X_test, axis = 1)
# X_test = np.append(wifi_t.ix[:, 'group'].reshape([749, 1]), X_test, axis = 1)
# X_test = np.append(np.array(range(0, 749)).reshape([749, 1]), X_test, axis = 1)
# # X_test = pd.DataFrame(X_test)
# # X_test = X_test.fillna(0)
# # X_test = X_test.as_matrix()
# print(X_test.shape)

#对于ap的k个分组
print('train...')
for k in range(0, clus_num):
    #train k th predictor:
	#按分组割开数据，并形成特征和样本标签
    X_t = X[X[:, 0] == k]
    train_X = X_t[:,1 : -18]
    train_Y = X_t[:, -18 : ]
    # train_X = pd.DataFrame(train_X)
    # train_X = train_X.fillna(train_X.mean())
    # train_Y = pd.DataFrame(train_Y)
    # train_Y = train_X.fillna(train_Y.mean())
    print(train_X.shape)
    print(train_Y.shape)
    clf = linear_model.MultiTaskElasticNet(max_iter=10000)
    clf.fit(train_X, train_Y)
    #score = cross_validation.cross_val_score(clf, train_X, train_Y, cv = 5, scoring = 'mean_squared_error')
    #print("time: " + str(k) + " Multi_Elastic:" + str(score.mean()))
    #f_score[k] = score.mean() * train_X.shape[0]

    # #predict:
    # #预测
    # X_t = X_test[X_test[:, 1] == k]
    # index = X_t[:, 0]
    # #X_t = pd.DataFrame(X_t)
    # #X_t = X_t.fillna(X_t.mean())
    # #X_t = X_t.as_matrix()
    # if k != 0:
    #     res_t = clf.predict(X_t[:, 2:])
    #     res_t = np.append(index.reshape(X_t.shape[0], 1), res_t, axis = 1)
    #     res = np.append(res_t, res, axis = 0)
    # else:
    #     #对于第一条数据特殊处理
    #     res = clf.predict(X_t[:, 2:])
    #     res = np.append(index.reshape(X_t.shape[0], 1), res, axis = 1)

#print("all score:" + str(f_score.sum() / X.shape[0]))
# res = pd.DataFrame(res)
# #按AP排序
# res = res.sort(0, axis = 0)
# res = res.drop([0], axis = 1).as_matrix()
# #将预测值为负的取0
# res[res < 0] = 0

#print("test error:" + str(np.power((res - wifi_t.ix[:, i: i + 18]), 2).sum().sum()))
#
# #形成提交结果
# sub = pd.read_csv('data/sub_template.csv')
# res = res.reshape(1, res.shape[0] * res.shape[1])
# sub['passengerCount'] = res[0]
# sub.to_csv('sub1_Mix_MultiElastic.csv', index = False)
