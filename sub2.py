import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor

wifi = pd.read_csv('data/wifi_AP_2D.csv')
wifi_t = wifi.drop(['WIFIAPTag'], axis = 1)
for i in [x for x in range(144, 553) if (x % 20 == 0) & (x != 0)]:
    if i not in [200, 320, 340, 360, 480, 500]:
        try:
            X1 = wifi_t.ix[:, i - 3 : i + 1]
            X1 = np.append(wifi_t.ix[:, i - 144 - 2 : i - 144 + 3], X1, axis = 1)
            X = np.append(X, X1, axis = 0)
        except:
            X = wifi_t.ix[:, i - 3 : i + 1]
            X = np.append(wifi_t.ix[:, i - 144 - 2 : i - 144 + 3], X, axis = 1)
print(X.shape)
X = np.append(X[:, 1 : 8] - X[:, : 7], X, axis = 1)
print(X.shape)

train_X = X[:, : 15]
train_Y = X[:, 15]
print(train_X.shape)

clf = linear_model.Lasso()
clf.fit(train_X, train_Y)
score = cross_validation.cross_val_score(clf, train_X, train_Y, cv = 10, scoring = 'mean_squared_error')
print("lasso:" + str(score.mean()))
print('feature importance:')
print(clf.coef_)

clf = linear_model.LinearRegression()
clf.fit(train_X, train_Y)
score = cross_validation.cross_val_score(clf, train_X, train_Y, cv = 10, scoring = 'mean_squared_error')
print("LR:" + str(score.mean()))
print('feature importance:')
print(clf.coef_)

clf = KernelRidge(alpha=1.0)
clf.fit(train_X, train_Y)
score = cross_validation.cross_val_score(clf, train_X, train_Y, cv = 10, scoring = 'mean_squared_error')
print("KR:" + str(score.mean()))
#print(clf.coef0)

clf = GradientBoostingRegressor(n_estimators = 100, learning_rate=0.1, subsample = 0.6)
clf.fit(train_X, train_Y)
score = cross_validation.cross_val_score(clf, train_X, train_Y, cv = 10, scoring = 'mean_squared_error')
print("GBDT:" + str(score.mean()))
print('feature importance:')
print(clf.feature_importances_)

# #predict
# for i in range(553, 571):
#     X = wifi_t.ix[:, i - 3 : i]
#     X = np.append(wifi_t.ix[:, i - 144 - 3 : i - 144 + 2], X, axis = 1)
#     print(X.shape)
#     X = np.append(X[:, 1 : 8] - X[:, : 7], X, axis = 1)
#     print(X.shape)
#     wifi_t[i] = clf.predict(X)
#
# #submit
# res = wifi_t.ix[:, 553:]
# sub = pd.read_csv('sub_template.csv')
# res = res.as_matrix().reshape(1, res.shape[0] * res.shape[1])
# sub['passengerCount'] = res[0]
# sub.to_csv('sub2_LinearRegrssion.csv', index = False)


