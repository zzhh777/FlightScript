import pandas as pd
import numpy as np

wifi = pd.read_csv('data/wifi_AP_2D.csv')
wifi = wifi.drop(['WIFIAPTag'], axis = 1)
wifi1 = wifi.ix[:, :144].as_matrix()
wifi2 = wifi.ix[:,144 :288].as_matrix()
wifi3 = wifi.ix[:,288 :288 + 144].as_matrix()
wifi = (wifi1 + wifi2 + wifi3) / 3
wifi = np.append(wifi, wifi, axis = 1)
wifi = np.append(wifi, wifi, axis = 1)
wifi = np.append(wifi, wifi, axis = 1)

wifi = pd.DataFrame(wifi)
wifi.to_csv('data/mean_in_day.csv', index = False)