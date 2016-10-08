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
            X1 = wifi_t.ix[:, i - 6 : i]
            X1 = np.append(wifi_t.ix[:, i - 144 - 3 : i - 144 + 4], X1, axis = 1)
            X = np.append(X, X1, axis = 0)
        except:
            X = wifi_t.ix[:, i - 6 : i]
            X = np.append(wifi_t.ix[:, i - 144 - 3 : i - 144 + 4], X, axis = 1)

train_X = X[:, : 12]
train_Y = X[:, 12]
print(train_X.shape)

# clf = linear_model.LinearRegression()
# clf.fit(train_X, train_Y)
# score = cross_validation.cross_val_score(clf, train_X, train_Y, cv = 5)
# print(clf.coef_)
# print("LR:" + str(score.mean()))
#
# clf = KernelRidge(alpha=1.0)
# clf.fit(train_X, train_Y)
# score = cross_validation.cross_val_score(clf, train_X, train_Y, cv = 5)
# print(clf.coef0)
# print("KR:" + str(score.mean()))
#
clf = GradientBoostingRegressor(n_estimators = 100, learning_rate=0.1)
clf.fit(train_X, train_Y)
# score = cross_validation.cross_val_score(clf, train_X, train_Y, cv = 5)
# print(clf.feature_importances_)
# print("GBDT:" + str(score.mean()))

#predict
for i in range(553, 571):
    print(i)
    X = wifi_t.ix[:, i - 5 : i + 1]
    X = np.append(wifi_t.ix[:, i - 144 - 2 : i - 144 + 5], X, axis = 1)
    wifi_t[i] = clf.predict(X)

#submit
res = wifi_t.ix[:, 553:]
sub = pd.read_csv('sub_template.csv')
res = res.as_matrix().reshape(1, res.shape[0] * res.shape[1])
sub['passengerCount'] = res[0]
sub.to_csv('sub1_GBDT.csv', index = False)


