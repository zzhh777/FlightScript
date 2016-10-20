import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#平均连接序列和对应长度的均值序列相关系数
wifi = pd.read_csv('data/wifi_AP_2D.csv')
wifi_mean = pd.read_csv('data/mean_in_day.csv')
wifi = wifi.drop(['WIFIAPTag'], axis = 1)
wifi_mean = wifi_mean.ix[:, :2130]
co = np.zeros(749)
for i in range(749):
    c = np.append([wifi.ix[i, :].as_matrix()], [wifi_mean.ix[i, :].as_matrix()], axis = 0)
    co[i] = np.corrcoef(c)[0, 1]

#天与天之间相关性分析,11号0点00是第30个时间片
wifi = pd.read_csv('data/wifi_AP_2D.csv')
miss = ['E2-3B<E2-3-09>','E2-3B<E2-3-10>','E2-3B<E2-3-11>','E2-3B<E2-3-12>',\
        'E2-3B<E2-3-13>','E2-3B<E2-3-14>','E2-3B<E2-3-15>','E2-3B<E2-3-16>']
wifi = wifi[wifi['WIFIAPTag'].apply(lambda x:x not in miss)]
wifi = wifi.drop(['WIFIAPTag'], axis = 1)
wifi.columns = range(2130)
day_data = np.zeros((14, 741 * 144))
for i in range(14):
    day_data[i, :] = wifi.iloc[:, 30 + 144 * i : 30 + 144 * (i + 1)].as_matrix().reshape(1, 741 * 144)
day_data = pd.DataFrame(day_data)
co = day_data.T.corr()
plt.imshow(co)

#sub refine
sub = pd.read_csv('rule_res.csv')
sub['passengerCount'] = sub['passengerCount'].apply(lambda x : '%.3f' % x)
sub.to_csv('sub2_Mix_MultiElastic.csv', index = False)

#单个WIFI天与天之间的联系
wifi = pd.read_csv('data/wifi_AP_2D.csv')
wifi = wifi.fillna(0)
Tag = wifi['WIFIAPTag']
wifi = wifi.drop(['WIFIAPTag'], axis = 1)
wifi_mean = pd.read_csv('data/mean_in_day.csv')
wifi_mean = wifi_mean.ix[:, :2130]
def refine_time(x):
    s = list(x)
    s[-2] = ':'
    x = ''.join(s)
    x = x + '0'
    return x
wifi.columns = pd.Series(wifi.columns).apply(refine_time).apply(lambda x : pd.Timestamp(x))
wifi_mean.columns = wifi.columns
#只留下co表现差的AP
co = pd.Series(co).replace('nan', np.nan)
co = co.fillna(0)
wifi = wifi[co < 0.5]
wifi.index = range(wifi.__len__())
wifi_mean = wifi_mean[co < 0.5]
wifi_mean.index = range(wifi_mean.__len__())

AP_id = 10
plt.subplot(211)
plt.plot(wifi.ix[AP_id, '2016-09-15 00:00:00':])
plt.plot(wifi_mean.ix[AP_id, '2016-09-15 00:00:00':])
#plt.plot(wifi.ix[AP_id, :])
#plt.plot(wifi_mean.ix[AP_id, :])
c = np.append([wifi.ix[AP_id, '2016-09-15 00:00:00':].as_matrix()],\
              [wifi_mean.ix[AP_id, '2016-09-15 00:00:00':].as_matrix()], axis = 0)
print('corelation:' + str(np.corrcoef(c)[0, 1]))

day_data = np.zeros((14, 144))
for i in range(14):
    day_data[i, :] = wifi.iloc[AP_id, 30 + 144 * i : 30 + 144 * (i + 1)].as_matrix().reshape(1, 144)
day_data = pd.DataFrame(day_data)
co_slice_and_mean = day_data.T.corr()
plt.subplot(212)
plt.imshow(co_slice_and_mean)

#与均值低相关的AP与飞机、乘客相关度分析
wifi = pd.read_csv('data/wifi_AP_2D.csv')
fly = pd.read_csv('data/fly_2D.csv')
gate = pd.read_csv('data/airport_gz_gates.csv')
fly = fly.merge(gate).drop(['BGATE_ID'], axis = 1)
fly.ix[:, : -1] = fly.groupby('BGATE_AREA').transform('sum')
fly = fly.drop_duplicates()

#此处co由平均连接序列和对应长度的均值序列相关系数预先计算得出
co = pd.Series(co).replace('nan', np.nan)
co = co.fillna(0)
wifi = wifi[co < 0.5]

wifi.index = wifi['WIFIAPTag']
fly.index = fly['BGATE_AREA']
wifi = wifi.drop(['WIFIAPTag'], axis = 1)
fly = fly.drop(['BGATE_AREA'], axis = 1)

union_index = np.array(list(set(wifi.columns) & set(fly.columns)))
fly = fly[union_index]
wifi = wifi[union_index]
wifi = wifi.fillna(0)
wifi_dis = wifi.copy()

for i in range(1, wifi.shape[1]):
    wifi_dis.ix[:, i] = wifi.ix[:, i] - wifi.ix[:, i - 1]
wifi_dis.ix[:, 0] = 0

comp = pd.concat([fly, wifi_dis])
co_Gate_and_AP = comp.T.corr()
plt.imshow(co_Gate_and_AP)

#单个wifi流量查看
wifi = pd.read_csv('data/wifi_AP_2D.csv')
Tag = wifi['WIFIAPTag']
wifi = wifi.drop(['WIFIAPTag'], axis = 1)
co = pd.Series(co).replace('nan', np.nan)
co = co.fillna(0)
wifi = wifi[co < 0.5]
def refine_time(x):
    s = list(x)
    s[-2] = ':'
    x = ''.join(s)
    x = x + '0'
    return x
wifi.columns = pd.Series(wifi.columns).apply(refine_time).apply(lambda x : pd.Timestamp(x))
wifi.index = range(wifi.__len__())
for i in range(5):
    plt.plot(wifi.ix[i, '2016-09-22 00:00:00':])

#