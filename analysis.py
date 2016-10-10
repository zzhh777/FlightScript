import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

wifi = pd.read_csv('data/wifi_AP_2D.csv')
check = pd.read_csv('data/depart_2D_ID.csv')

check.index = check.index.astype(str)
wifi = wifi.drop(['WIFIAPTag'], axis = 1)
wifi2 = np.append(wifi.as_matrix(), check.ix[3:556, :].T.as_matrix(), axis = 0)
print(wifi2.shape)
wifi2 = pd.DataFrame(wifi2)
#wifi2 = wifi2.drop(['WIFIAPTag'], axis = 1)
c = wifi2.T.corr()
plt.imshow(c)