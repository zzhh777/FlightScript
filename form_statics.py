import pandas as pd
import numpy as np

def group_WIFI(x):
    x[0] = x[0].replace('<', '-')
    s = x[0].split('-')
    x[0] = s[0] + s[1]
    return x

wifi = pd.read_csv('data/wifi_AP_2D.csv')
wifi_t = wifi.drop(['WIFIAPTag'], axis = 1)
tag = wifi[['WIFIAPTag']]
tag.apply(group_WIFI, axis = 1)
wifi['group'] = tag['WIFIAPTag']

#mean
wifi_mean_g = wifi.groupby('group').transform('mean')
try:
    wifi_mean_g = wifi_mean_g.drop(['WIFIAPTag'], axis = 1)
except:
    pass
wifi_mean_g.to_csv('data/static_mean.csv', index = False)

#sum
wifi_sum_g = wifi.groupby('group').transform('sum')
try:
    wifi_sum_g = wifi_sum_g.drop(['WIFIAPTag'], axis = 1)
except:
    pass
wifi_sum_g.to_csv('data/static_sum.csv', index = False)

#var
wifi_var_g = wifi.groupby('group').transform('var')
try:
    wifi_var_g = wifi_var_g.drop(['WIFIAPTag'], axis = 1)
except:
    pass
wifi_var_g = wifi_var_g.fillna(0)
wifi_var_g.to_csv('data/static_var.csv', index = False)

#max
wifi_max_g = wifi.groupby('group').transform('max')
try:
    wifi_max_g = wifi_max_g.drop(['WIFIAPTag'], axis = 1)
except:
    pass
wifi_max_g.to_csv('data/static_max.csv', index = False)

#min
wifi_min_g = wifi.groupby('group').transform('min')
try:
    wifi_min_g = wifi_min_g.drop(['WIFIAPTag'], axis = 1)
except:
    pass
wifi_min_g.to_csv('data/static_min.csv', index = False)
