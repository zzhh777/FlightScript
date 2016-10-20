import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor

#import data and init variance
print('reading data...')
link_slice_10 = pd.read_csv('data/wifi_AP_2D.csv')
link_slice_mean = pd.read_csv('data/mean_in_day.csv')

link_slice_10 = link_slice_10.drop(['WIFIAPTag'], axis = 1)
link_slice_mean = link_slice_mean.ix[:, :2130]
link_slice_10.columns = range(2130)
link_slice_mean.columns = range(2130)

data_for_cluster = link_slice_mean.copy()
############################
#连接值和均值分别求滑动均值#
############################
print('getting feature...')
window_size = 3
#AP均值的一阶滑动均值
mean_slide_mean_1 = np.zeros((749, 144))
for i in range(144):
    mean_slide_mean_1[:, i] = link_slice_mean.iloc[:, i - window_size + 144: i + 144].T.mean().T
#复制4遍为之后的计算做准备
mean_slide_mean_1_temp = np.append(mean_slide_mean_1, mean_slide_mean_1, axis = 1)
mean_slide_mean_1_temp = np.append(mean_slide_mean_1_temp, mean_slide_mean_1_temp, axis = 1)
mean_slide_mean_1_temp = pd.DataFrame(mean_slide_mean_1_temp)
mean_slide_mean_1 = pd.DataFrame(mean_slide_mean_1)

#AP均值的二阶滑动均值
mean_slide_mean_2 = np.zeros((749, 144))
for i in range(144):
    mean_slide_mean_2[:, i] = mean_slide_mean_1_temp.iloc[:, i - window_size + 144: i + 144].T.mean().T
mean_slide_mean_2 = pd.DataFrame(mean_slide_mean_2)

#AP均值的一阶差分
mean_One_order_diff = link_slice_mean.iloc[:, 144 : 288].as_matrix() - link_slice_mean.iloc[:,143 : 287].as_matrix()
mean_One_order_diff = pd.DataFrame(mean_One_order_diff)
#复制4遍为之后的计算做准备
mean_One_order_diff_temp = np.append(mean_One_order_diff, mean_One_order_diff, axis = 1)
mean_One_order_diff_temp = np.append(mean_One_order_diff_temp, mean_One_order_diff_temp, axis = 1)
mean_One_order_diff_temp = pd.DataFrame(mean_One_order_diff_temp)

#AP均值的二阶差分
mean_Two_order_diff = mean_One_order_diff_temp.iloc[:, 144 : 288].as_matrix() - mean_One_order_diff_temp.iloc[:,143 : 287].as_matrix()
mean_Two_order_diff = pd.DataFrame(mean_Two_order_diff)

