import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor

wifi = pd.read_csv('data/wifi_AP_2D.csv')
wifi_t = wifi.drop(['WIFIAPTag'], axis = 1)
depart = pd.read_csv('data/depart_2D.csv')
s_check = pd.read_csv('data/s_check_2D.csv')


#for i in [x for x in range(144, 553) if (x in range(409, 426)) | (x in range(256, 282)) | ((x % 20 == 0) & (x != 0))]:
for i in [x for x in range(144, 553) if ((x % 20 == 0) & (x != 0))]:
    if i not in [200, 320, 340, 360, 480, 500]:    
#        if (i != 256):
        if(i != 160):
            X1 = wifi_t.ix[:, i - 2 : i + 1]
            X1 = np.append(wifi_t.ix[:, i - 144 - 1 : i - 144 + 2], X1, axis = 1)
            X1 = np.append(np.tile(list(depart.T.sum()[i - 1 : i + 2] / 749), (749, 1)), X1, axis = 1)
            X1 = np.append(np.tile(list(s_check.T.sum()[i - 1 : i + 2] / 749), (749, 1)), X1, axis = 1)
            X = np.append(X, X1, axis = 0)
        else:
            X = wifi_t.ix[:, i - 2 : i + 1]
            X = np.append(wifi_t.ix[:, i - 144 - 1 : i - 144 + 2], X, axis = 1)
            X = np.append(np.tile(list(depart.T.sum()[i - 1 : i + 2] / 749), (749, 1)), X, axis = 1)
            X = np.append(np.tile(list(s_check.T.sum()[i - 1 : i + 2] / 749), (749, 1)), X, axis = 1)
print(X.shape)
train_X = X[:, : 11]
train_Y = X[:, 11]


clf = linear_model.Ridge (alpha = 0.5)
clf.fit(train_X, train_Y)
#score = cross_validation.cross_val_score(clf, train_X, train_Y, cv = 5, scoring = 'mean_squared_error')
#print("RR:" + str(score.mean()))
#print('feature importance:')
#print(clf.coef_)

#clf = linear_model.LinearRegression()
#clf.fit(train_X, train_Y)
#score = cross_validation.cross_val_score(clf, train_X, train_Y, cv = 5, scoring = 'mean_squared_error')
#print("LR:" + str(score.mean()))
#print('feature importance:')
#print(clf.coef_)

#bag = BaggingRegressor(linear_model.LinearRegression())
#bag.fit(train_X, train_Y)
#score = cross_validation.cross_val_score(clf, train_X, train_Y, cv = 5, scoring = 'mean_squared_error')
#print("bagging:" + str(score.mean()))
#print('feature importance:')
#print(clf.coef_)

#clf = linear_model.Lasso()
#clf.fit(train_X, train_Y)
#score = cross_validation.cross_val_score(clf, train_X, train_Y, cv = 10, scoring = 'mean_squared_error')
#print("lasso:" + str(score.mean()))
#print('feature importance:')
#print(clf.coef_)

#clf = GradientBoostingRegressor(n_estimators = 100, learning_rate=0.1)
#clf.fit(train_X, train_Y)
#score = cross_validation.cross_val_score(clf, train_X, train_Y, cv = 5, scoring = 'mean_squared_error')
#print("GBDT:" + str(score.mean()))
#print('feature importance:')
#print(clf.feature_importances_)

#clf = RandomForestRegressor(n_estimators = 100)
#clf.fit(train_X, train_Y)
#score = cross_validation.cross_val_score(clf, train_X, train_Y, cv = 5, scoring = 'mean_squared_error')
#print("RF:" + str(score.mean()))
#print('feature importance:')
#print(clf.feature_importances_)

#predict
for i in range(553, 571):
    X = wifi_t.ix[:, i - 2 : i]
    X = np.append(wifi_t.ix[:, i - 144 - 1 : i - 144 + 2], X, axis = 1)
    X = np.append(np.tile(list(depart.T.sum()[i - 1 : i + 2] / 749), (749, 1)), X, axis = 1)
    X = np.append(np.tile(list(s_check.T.sum()[i - 1 : i + 2] / 749), (749, 1)), X, axis = 1)
    wifi_t[i] = clf.predict(X)

#submit
res = wifi_t.ix[:, 553:]
sub = pd.read_csv('sub_template.csv')
res = res.as_matrix().reshape(1, res.shape[0] * res.shape[1])
sub['passengerCount'] = res[0]
sub.to_csv('sub3_RidgeRegression.csv', index = False)


