import pandas as pd
import numpy as np

wifi = pd.read_csv('data/wifi_AP_2D.csv')
wifi = wifi.drop(['WIFIAPTag'], axis = 1)
wifi.columns = range(2130)
wifi_mean = np.zeros((749, 144))
for i in range(144):
    #1471
    index = [x for x in range(1950) if (x % 144 == i)]
    wifi_temp = wifi.ix[:, index]
    wifi_temp = wifi_temp.T.mean()
    wifi_mean[:, i] = wifi_temp.as_matrix()
for i in range(4):
    wifi_mean = np.append(wifi_mean, wifi_mean, axis = 1)

wifi_mean = pd.DataFrame(wifi_mean)
wifi_mean.to_csv('data/mean_in_day.csv', index = False)