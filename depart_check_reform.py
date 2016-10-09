import pandas as pd
import numpy as np

def fly_ref(x):
    try:
        x[1] = int(x[1]) + 48
    except:
        pass
    try:
        x[2] = int(x[2]) + 48
    except:
        pass
    if type(x[3]) == str:
        x[3].replace('"', '')
        x[3] = x[3].split(',')[0]
    return x

def real_time(x):
    try:
        x[0] = int(x[2])
    except:
        pass
    return x

def real_area(x):
    try:
        if (x[1] - x[0] > 0) & (x[1] - x[0] < 30):
            return 1
        else:
            return 0
    except:
        return 0
#get 2-D tables
fly = pd.read_csv('data/my_fly.csv')
fly = fly.apply(fly_ref, axis = 1)

gate = pd.read_csv('data/gates.csv')
fly = fly.merge(gate, how = 'left')
fly = fly.drop(['BGATE_ID'], axis = 1)

#get depart 2-D table
depart = pd.read_csv('data/my_depart.csv')
depart = depart.drop(['checkin_time'], axis = 1)
depart = depart.dropna()
depart['time'] = depart.flight_time.astype(int)

fly['time'] = fly['scheduled_flt_time']
depart = depart.merge(fly, how = 'left', on = ['time', 'flight_ID'])
depart = depart[['time', 'BGATE_AREA', 'actual_flt_time']]
depart = depart.apply(real_time, axis = 1)
depart = depart.drop(['actual_flt_time'], axis = 1)
depart = depart.replace({'E1' : 0, 'E2' : 1, 'E3' : 2, 'W1' : 3, 'W2' : 4, 'W3' : 5,})

depart_2D = np.zeros([1000, 7])
for i in range(0, depart.__len__()):
    x = depart.iloc[i][0]
    y = depart.iloc[i][1]
    if np.isnan(depart.iloc[i][1]):
        depart_2D[x][6] = depart_2D[x][6] + 1
    else:
        depart_2D[x][int(y)] = depart_2D[x][int(y)] + 1
depart_2D = pd.DataFrame(depart_2D)
depart_2D.to_csv('data/depart_2D.csv', index = False)

#get safe_check 2-D table
s_check = pd.read_csv('data/my_check.csv')
s_check = s_check.merge(fly[['flight_ID', 'time', 'BGATE_AREA']])
s_check = s_check[['security_time', 'time', 'BGATE_AREA']]
s_check['res'] = s_check.apply(real_area, axis = 1)
s_check = s_check[s_check['res'] == 1]
s_check = s_check.replace({'E1' : 0, 'E2' : 1, 'E3' : 2, 'W1' : 3, 'W2' : 4, 'W3' : 5,})
s_check = s_check[['security_time', 'BGATE_AREA']]

s_check_2D = np.zeros([1000, 7])
for i in range(0,  s_check.__len__()):
    x = s_check.iloc[i][0]
    y = s_check.iloc[i][1]
    if np.isnan(s_check.iloc[i][1]):
        s_check_2D[x][6] = s_check_2D[x][6] + 1
    else:
        s_check_2D[x][int(y)] = s_check_2D[x][int(y)] + 1
s_check_2D = pd.DataFrame(s_check_2D)
s_check_2D.to_csv('data/s_check_2D.csv', index = False)