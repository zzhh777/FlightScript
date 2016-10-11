import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor

wifi = pd.read_csv('data/wifi_AP_2D.csv')
wifi_t = wifi.drop(['WIFIAPTag'], axis = 1)
#wifi_t = wifi
g_max = pd.read_csv('data/static_max.csv')
g_mean = pd.read_csv('data/static_mean.csv')
g_min = pd.read_csv('data/static_min.csv')
g_sum = pd.read_csv('data/static_sum.csv')
g_var = pd.read_csv('data/static_var.csv')

for i in [x for x in range(144, 553) if (x % 20 == 0) & (x != 0)]:
    #print(i)
    if i not in [200, 320, 340, 360, 480, 500, 540]:
        if i != 160:
            X1 = wifi_t.ix[:, i - 3 : i + 18]
            X1 = np.append(wifi_t.ix[:, i - 144 - 2 : i - 144 + 20], X1, axis = 1)
            X1 = np.append(g_max.ix[:, i - 2 : i], X1, axis = 1)
            X1 = np.append(g_max.ix[:, i - 144 - 2 : i - 144 + 20], X1, axis = 1)
            X1 = np.append(g_mean.ix[:, i - 2 : i], X1, axis = 1)
            X1 = np.append(g_mean.ix[:, i - 144 - 2 : i - 144 + 20], X1, axis = 1)
            X1 = np.append(g_min.ix[:, i - 2 : i], X1, axis = 1)
            X1 = np.append(g_min.ix[:, i - 144 - 2 : i - 144 + 20], X1, axis = 1)
            X1 = np.append(g_sum.ix[:, i - 2 : i], X1, axis = 1)
            X1 = np.append(g_sum.ix[:, i - 144 - 2 : i - 144 + 20], X1, axis = 1)
            X1 = np.append(g_var.ix[:, i - 2 : i], X1, axis = 1)
            X1 = np.append(g_var.ix[:, i - 144 - 2 : i - 144 + 20], X1, axis = 1)
            X = np.append(X, X1, axis = 0)
        else:
            X = wifi_t.ix[:, i - 3 : i + 18]
            X = np.append(wifi_t.ix[:, i - 144 - 2 : i - 144 + 20], X, axis = 1)
            X = np.append(g_max.ix[:, i - 2 : i], X, axis = 1)
            X = np.append(g_max.ix[:, i - 144 - 2 : i - 144 + 20], X, axis = 1)
            X = np.append(g_mean.ix[:, i - 2 : i], X, axis = 1)
            X = np.append(g_mean.ix[:, i - 144 - 2 : i - 144 + 20], X, axis = 1)
            X = np.append(g_min.ix[:, i - 2 : i], X, axis = 1)
            X = np.append(g_min.ix[:, i - 144 - 2 : i - 144 + 20], X, axis = 1)
            X = np.append(g_sum.ix[:, i - 2 : i], X, axis = 1)
            X = np.append(g_sum.ix[:, i - 144 - 2 : i - 144 + 20], X, axis = 1)
            X = np.append(g_var.ix[:, i - 2 : i], X, axis = 1)
            X = np.append(g_var.ix[:, i - 144 - 2 : i - 144 + 20], X, axis = 1)

print(X.shape)

train_X = X[:, : 145]
train_Y = X[:, 145 : ]
print(train_X.shape)
print(train_Y.shape)

clf = linear_model.MultiTaskElasticNet(max_iter=10000)
clf.fit(train_X, train_Y)
score = cross_validation.cross_val_score(clf, train_X, train_Y, cv = 10, scoring = 'mean_squared_error')
print("Multi_Elastic:" + str(score.mean() * 749 * 18))
#print('feature importance:')
#print(clf.coef_)

clf = linear_model.MultiTaskLasso(max_iter=10000)
clf.fit(train_X, train_Y)
score = cross_validation.cross_val_score(clf, train_X, train_Y, cv = 10, scoring = 'mean_squared_error')
print("Multi_Lasso:" + str(score.mean() * 749 * 18))
#print('feature importance:')
#print(clf.coef_)



#predict
# i = 553
# X = wifi_t.ix[:, i - 3 : i]
# X = np.append(wifi_t.ix[:, i - 144 - 2 : i - 144 + 20], X, axis = 1)
# X = np.append(g_max.ix[:, i - 2: i], X, axis = 1)
# X = np.append(g_mean.ix[:, i - 2 : i], X, axis = 1)
# X = np.append(g_min.ix[:, i - 2 : i], X, axis = 1)
# X = np.append(g_sum.ix[:, i - 2 : i], X, axis = 1)
# X = np.append(g_var.ix[:, i - 2 : i], X, axis = 1)
# res = clf.predict(X)
# print(res.shape)
#
# #submit
# sub = pd.read_csv('data/sub_template.csv')
# res = res.reshape(1, res.shape[0] * res.shape[1])
# sub['passengerCount'] = res[0]
# sub.to_csv('sub4_Multi_Lasso.csv', index = False)