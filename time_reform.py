#this is code record that can't run directly
import pandas as pd
import numpy as np

def get_time_dic():
    #read data
    dic_t = pd.read_csv('data/time_index.csv')
    dic = {}
    for i in range(0, dic_t.__len__()):
        dic[dic_t.ix[i][0]] = dic_t.ix[i][1]
    return dic

def replace_time(x):
    if type(x[1]) == np.str:
        s = list(x[1])
        s.insert(5, '0')
        x[1] = ''.join(s)
        x[1] = x[1].replace('/', ' ').replace(':', ' ').replace(' ', '-')[:-4]
    if type(x[2]) == np.str:
        s = list(x[2])
        s.insert(5, '0')
        x[2] = ''.join(s)
        x[2] = x[2].replace('/', ' ').replace(':', ' ').replace(' ', '-')[:-4]
    return x

#transform time data into int and saving
print('reading data...')
depart = pd.read_csv('data/departure.csv')
print('change time form...')
depart = depart.apply(replace_time, axis = 1)
print('getting time dic...')
dic = get_time_dic()
print('replace timedata...')
depart = depart.replace(dic)
depart.to_csv('data/my_depart.csv', index = False)

print('reading data...')
s_check = pd.read_csv('data/security_check.csv')
print('change time form...')
s_check = s_check.apply(replace_time, axis = 1)
print('getting time dic...')
dic = get_time_dic()
print('replace timedata...')
s_check = s_check.replace(dic)
s_check.to_csv('data/my_s_check.csv', index = False)


print('reading data...')
fly = pd.read_csv('data/flights.csv')
print('change time form...')
fly = fly.apply(replace_time, axis = 1)
print('getting time dic...')
dic = get_time_dic()
print('replace timedata...')
fly = fly.replace(dic)
fly.to_csv('data/my_fly.csv', index = False)