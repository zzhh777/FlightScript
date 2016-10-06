import pandas as pd

#read data
wifi = pd.read_csv('data/wifi_ap.csv')
#combine time into 10min and replace it as and int that ordered by time
wifi['timeStamp'] = wifi['timeStamp'].apply(lambda x:x[:-4])
timeStamp = wifi['timeStamp'].drop_duplicates()
dic = {}
#replace time as int
k = 0
for i in timeStamp:
    dic[i] = k
    k = k + 1
wifi = wifi.replace(dic)

#transform the data into 2-D table that cols are time and row are WIFI_AP
AP = wifi['WIFIAPTag'].drop_duplicates()
wifi_2D = pd.DataFrame({'WIFIAPTag' : AP})
for i in range(0, 553):
    wifi_2D[i] = 0
AP_sum = wifi.groupby(['WIFIAPTag', 'timeStamp']).mean()
for i in range(0, wifi_2D.__len__()):
    print(i)
    for j in range(0, 553):
        try:
            wifi_2D.ix[i, j] = AP_sum.loc[wifi_2D.ix[i, 'WIFIAPTag']].loc[j][0]
        except:
            pass
wifi_2D.to_csv('data/wifi_AP_2D.csv', index = False)