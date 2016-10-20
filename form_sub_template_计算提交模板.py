import pandas as pd
import numpy as np

wifi = pd.read_csv('data/wifi_AP_2D.csv')
AP = wifi['WIFIAPTag']
# year = 2016
# month = 9
# day = 25
hour = 15
m = 0
times = np.array([])
for i in range(18):
    times = np.append(times, '2016-09-25-' + str(hour) + '-' + str(m))
    m = m + 1
    if m == 6:
        hour = hour + 1
        m = 0
sub_table = np.zeros((18 * AP.__len__(), 2))
sub_table = pd.DataFrame(sub_table).astype(str)
k = 0
for ap in AP:
    for time in times:
        sub_table.ix[k][0] = ap
        sub_table.ix[k][1] = time
        k = k + 1
sub_table.columns = ['WIFIAPTag', 'slice10min']
sub_table.to_csv('data/sub_template.csv', index = False)