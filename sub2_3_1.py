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

#init data
link_slice_10 = pd.read_csv('data/wifi_AP_2D.csv')
link_slice_mean = pd.read_csv('data/mean_in_day.csv')

link_slice_10 = link_slice_10.drop(['WIFIAPTag'], axis = 1)
link_slice_mean = link_slice_mean.ix[:, :2130]
link_slice_10.columns = range(2130)
link_slice_mean.columns = range(2130)
############################
#连接值和均值分别求滑动均值#
############################
print('getting feature...')
window_size = 3
#一阶滑动均值
#连接数的一阶滑动均值
slide_mean_1 = np.zeros((link_slice_10.shape[0], link_slice_10.shape[1]))
for i in range(window_size, link_slice_10.shape[1]):
    slide_mean_1[:, i] = link_slice_10.iloc[:, i - window_size : i].T.mean().T
slide_mean_1 = pd.DataFrame(slide_mean_1)
#AP均值的一阶滑动均值
mean_slide_mean_1 = np.zeros((749, 144))
for i in range(144):
    mean_slide_mean_1[:, i] = link_slice_mean.iloc[:, i - window_size + 144: i + 144].T.mean().T
#复制4遍为之后的计算做准备
mean_slide_mean_1_temp = np.append(mean_slide_mean_1, mean_slide_mean_1, axis = 1)
mean_slide_mean_1_temp = np.append(mean_slide_mean_1_temp, mean_slide_mean_1_temp, axis = 1)
mean_slide_mean_1_temp = pd.DataFrame(mean_slide_mean_1_temp)
mean_slide_mean_1 = pd.DataFrame(mean_slide_mean_1)

#二阶滑动均值
#AP连接数的二阶滑动均值
slide_mean_2 = np.zeros((link_slice_10.shape[0], link_slice_10.shape[1]))
for i in range(window_size, link_slice_10.shape[1]):
    slide_mean_2[:, i] = slide_mean_1.iloc[:, i - window_size : i].T.mean().T
slide_mean_2 = pd.DataFrame(slide_mean_2)
#AP均值的二阶滑动均值
mean_slide_mean_2 = np.zeros((749, 144))
for i in range(144):
    mean_slide_mean_2[:, i] = mean_slide_mean_1_temp.iloc[:, i - window_size + 144: i + 144].T.mean().T
mean_slide_mean_2_temp = np.append(mean_slide_mean_2, mean_slide_mean_2, axis = 1)
mean_slide_mean_2_temp = np.append(mean_slide_mean_2_temp, mean_slide_mean_2_temp, axis = 1)
mean_slide_mean_2_temp = pd.DataFrame(mean_slide_mean_2_temp)
mean_slide_mean_2 = pd.DataFrame(mean_slide_mean_2)

#一阶差分
#AP连接数的一阶差分
link_One_order_diff = np.zeros((link_slice_10.shape[0], link_slice_10.shape[1]))
link_One_order_diff[:, 1:] = link_slice_10.iloc[:, 1 :].as_matrix() - link_slice_10.iloc[:, : -1].as_matrix()
link_One_order_diff = pd.DataFrame(link_One_order_diff)
#AP均值的一阶差分
mean_One_order_diff = link_slice_mean.iloc[:, 144 : 288].as_matrix() - link_slice_mean.iloc[:,143 : 287].as_matrix()
mean_One_order_diff = pd.DataFrame(mean_One_order_diff)
#复制4遍为之后的计算做准备
mean_One_order_diff_temp = np.append(mean_One_order_diff, mean_One_order_diff, axis = 1)
mean_One_order_diff_temp = np.append(mean_One_order_diff_temp, mean_One_order_diff_temp, axis = 1)
mean_One_order_diff_temp = pd.DataFrame(mean_One_order_diff_temp)
#二阶差分
#AP连接数的二阶差分
link_Two_order_diff = np.zeros((link_slice_10.shape[0], link_slice_10.shape[1]))
link_Two_order_diff[:, 1:] = link_One_order_diff.iloc[:, 1 :].as_matrix() - link_One_order_diff.iloc[:, : -1].as_matrix()
link_Two_order_diff = pd.DataFrame(link_Two_order_diff)
#AP均值的二阶差分
mean_Two_order_diff = mean_One_order_diff_temp.iloc[:, 144 : 288].as_matrix() - mean_One_order_diff_temp.iloc[:,143 : 287].as_matrix()
mean_Two_order_diff_temp = np.append(mean_Two_order_diff, mean_Two_order_diff, axis = 1)
mean_Two_order_diff_temp = np.append(mean_Two_order_diff_temp, mean_Two_order_diff_temp, axis = 1)
mean_Two_order_diff_temp = np.append(mean_Two_order_diff_temp, mean_Two_order_diff_temp, axis = 1)
mean_Two_order_diff_temp = pd.DataFrame(mean_Two_order_diff_temp)

##############
#clusting data
#############
# #k-means聚类
# #normal cluster
# clus = KMeans(n_clusters = clus_num)
# clust_X = wifi.drop(['WIFIAPTag'], axis = 1).dropna(axis = 1)
# #clust_X = max_abs_scaler.fit_transform(clust_X)
# clus.fit(clust_X)
# wifi['group'] = clus.predict(clust_X)
# wifi_t['group'] = wifi['group']

#AP cluster
# #AP聚类
clus = AffinityPropagation(damping=0.98)
data_for_cluster = link_slice_mean.iloc[:, :144].copy()
link_slice_10['group'] = clus.fit_predict(data_for_cluster)
clus_num = link_slice_10['group'].max() + 1
print('cluster number:' + str(clus_num))

#get train data
#采集训练数据，每200分钟采集一次，
print('getting data...')
for i in [x for x in range(1470, 1950) if (x % 5 == 0) & (x != 0)]:
    #print(i)
    if i not in range(2100, 2130):
        #print(i)
        if ((i + 17) % 144) < (i % 144):
            last_i = (i + 17) % 144 + 144
        else:
            last_i = (i + 17) % 144
        if i != 1470:
            data_t = link_slice_10.ix[:, i - 3 : i + 18]
            data_t = np.append(link_slice_mean.ix[:, i - 2 : i + 19], data_t, axis = 1)
            #data_t = np.append(mean_One_order_diff_temp.ix[:, i % 144: last_i], data_t, axis = 1)
            data_t = np.append(mean_Two_order_diff_temp.ix[:, i % 144: last_i], data_t, axis = 1)
            data_t = np.append(link_slice_10[['group']].as_matrix(), data_t, axis = 1)
            data = np.append(data, data_t, axis = 0)
        else:
            data = link_slice_10.ix[:, i - 3 : i + 18]
            data = np.append(link_slice_mean.ix[:, i - 2 : i + 19], data, axis = 1)
            #data = np.append(mean_One_order_diff_temp.ix[:, i % 144 : last_i], data, axis = 1)
            data = np.append(mean_Two_order_diff_temp.ix[:, i % 144: last_i], data, axis = 1)
            data = np.append(link_slice_10[['group']].as_matrix(), data, axis = 1)

print(data.shape)
f_score = np.zeros(clus_num)
data = pd.DataFrame(data).dropna().as_matrix()

#形成测试集/预测目标
offline = True
i = 2130 - 18
if ((i + 17) % 144) < (i % 144):
    last_i = (i + 17) % 144 + 144
else:
    last_i = (i + 17) % 144
real_test = link_slice_10.ix[:, i - 3 : i]
real_test = np.append(link_slice_mean.ix[:, i - 144 - 2 : i - 144 + 19], real_test, axis = 1)
#real_test = np.append(mean_Two_order_diff_temp.ix[:, i % 144 : last_i], real_test, axis = 1)
real_test = np.append(mean_Two_order_diff_temp.ix[:, i % 144: last_i], real_test, axis = 1)
real_test = np.append(link_slice_10.ix[:, 'group'].reshape([749, 1]), real_test, axis = 1)

real_test = np.append(np.array(range(0, 749)).reshape([749, 1]), real_test, axis = 1)
real_test = pd.DataFrame(real_test).fillna(0).as_matrix()
print(real_test.shape)

if offline == True:
    data_train = data[:-10000, :]
    data_test = data[-10000:, :]
else:
    data_train = data
#对于ap的k个分组
print('train...')
for k in range(0, clus_num):
    #train k th predictor:
    data_t = data_train[data_train[:, 0] == k]
    train_X = data_t[:, 1 : -18]
    train_Y = data_t[:, -18:]
    print(train_X.shape)
    clf = linear_model.MultiTaskElasticNet(max_iter=10000)
    clf.fit(train_X, train_Y)

    if offline == True:
        data_t = data_test[data_test[:, 0] == k]
        test_X = data_t[:, 1 : -18]
        test_Y = data_t[:, -18 :]
        y_pred = clf.predict(test_X)
        #score = cross_validation.cross_val_score(clf, train_X, train_Y, cv = 5, scoring = 'mean_squared_error')
        score = np.power((test_Y - y_pred), 2)
        print("time: " + str(k) + " Multi_Elastic:" + str(score.mean()))
        f_score[k] = score.mean() * test_X.shape[0]

    #predict:
    #预测
    data = real_test[real_test[:, 1] == k]
    index = data[:, 0]
    if k != 0:
        res_t = clf.predict(data[:, 2:])
        res_t = np.append(index.reshape(data.shape[0], 1), res_t, axis = 1)
        res = np.append(res_t, res, axis = 0)
    else:
        res = clf.predict(data[:, 2:])
        res = np.append(index.reshape(data.shape[0], 1), res, axis = 1)

if offline == True:
    print("all score:" + str(f_score.sum() / data_test.shape[0]))

res = pd.DataFrame(res)
res = res.sort(0, axis = 0)
res = res.drop([0], axis = 1).as_matrix()
res[res < 0] = 0

if offline == True:
    print("test error:" + str(np.power((res - link_slice_10.ix[:, i: i + 18]), 2).sum().sum()))
else:
    #形成提交结果
    sub = pd.read_csv('data/sub_template.csv')
    res = res.reshape(1, res.shape[0] * res.shape[1])
    sub['passengerCount'] = res[0]
    sub = sub[['passengerCount', 'WIFIAPTag', 'slice10min']]
    sub.to_csv('sub4_Mix_MultiElastic.csv', index = False)
