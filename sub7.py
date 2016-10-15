import pandas as pd
import numpy as np
from sklearn import cross_validation
from keras.layers import Input, Embedding, Dense, merge, Flatten
from keras.models import Model
from keras.regularizers import l2

def group_WIFI(x):
    x[0] = x[0].replace('<', '-')
    s = x[0].split('-')
    x[0] = s[0] + s[1]
    return x

wifi = pd.read_csv('data/wifi_AP_2D.csv')
wifi_t = wifi.drop(['WIFIAPTag'], axis = 1)
tag = wifi[['WIFIAPTag']]
tag.apply(group_WIFI, axis = 1)
wifi_t['group'] = tag['WIFIAPTag']
tag = tag.drop_duplicates()
dic = dict(zip(tag['WIFIAPTag'], range(0, tag.__len__())))
wifi_t = wifi_t.replace(dic)
wifi_mean = pd.read_csv('data/mean_in_day.csv')

for i in [x for x in range(144, 553) if (x % 20 == 0) & (x != 0)]:
    #print(i)
    if i not in [540, 550]:
    #if i not in [200, 320, 340, 360, 480, 500, 540, 550]:
        if i != 160:
            X1 = wifi_t.ix[:, i - 3 : i + 18]
            X1 = np.append(wifi_mean.ix[:, i - 144 - 2 : i - 144 + 20], X1, axis = 1)
            X1 = np.append(np.tile(wifi_t.ix[:, i - 1].T, (749, 1)), X1, axis = 1)
            X1 = np.append(wifi_t.ix[:, 'group'].reshape([749, 1]), X1, axis = 1)
            X = np.append(X, X1, axis = 0)
        else:
            X = wifi_t.ix[:, i - 3 : i + 18]
            X = np.append(wifi_mean.ix[:, i - 144 - 2 : i - 144 + 20], X, axis = 1)
            X = np.append(np.tile(wifi_t.ix[:, i - 1].T, (749, 1)), X, axis = 1)
            X = np.append(wifi_t.ix[:, 'group'].reshape([749, 1]), X, axis = 1)

print(X.shape)
#np.random.shuffle(X)
train_X = X[:, : -18]
train_Y = X[:, -18 : ]
# train_X = X[0:-7000:, : -18]
# train_Y = X[0:-7000:, -18 : ]
# test_X = X[-7000:, : -18]
# test_Y = X[-7000:, -18 : ]
print(train_X.shape)
print(train_Y.shape)

print('compile..')
regular_factor = 0.3

G_input = Input(shape = (1, ), name = 'G_input')
G_embedding = Embedding(89, 10, input_length = 1)(G_input)
G_x = Flatten()(G_embedding)
#G_x = Dense(10, activation='relu')(G_x)
G_x = Dense(20, activation='relu', W_regularizer = l2(regular_factor))(G_x)

AP_input = Input(shape=(749, ), name='AP_input')
#AP_x = Dense(50, activation='relu')(AP_input)
AP_x = Dense(40, activation='relu', W_regularizer = l2(regular_factor))(AP_input)

his_input = Input(shape = (25, ), name = 'his_input')
#his_x = Dense(25, activation='relu')(his_input)
his_x = Dense(25, activation='relu', W_regularizer = l2(regular_factor))(his_input)

x = merge([AP_x, his_x, G_x], mode = 'concat')
#x = Dense(50, activation='relu')(x)
x = Dense(60, activation='relu', W_regularizer = l2(regular_factor))(x)
main_loss = Dense(18, activation='linear', name='main_output')(x)

model = Model(input=[AP_input, his_input, G_input], output=[main_loss])
model.compile(optimizer='rmsprop', loss='mse', loss_weights=[1])


model.fit({'G_input' : train_X[:, 0], 'AP_input' : train_X[:, 1 : 750], 'his_input' : train_X[:, 750 :]}, train_Y,\
    batch_size=60, nb_epoch=100, shuffle = True)
#model.fit({'G_input' : train_X[:, 0], 'AP_input' : train_X[:, 1 : 750], 'his_input' : train_X[:, 750 :]}, train_Y,\
#    batch_size=60, nb_epoch=100, validation_split = 0.2, shuffle = True)

# #validation
# f_score = np.zeros(10)
# for i in range(0, 10):
#     model.fit({'G_input' : train_X[:, 0], 'AP_input' : train_X[:, 1 : 750], 'his_input' : train_X[:, 750 :]}, train_Y,\
#               batch_size=60, nb_epoch=10, validation_split = 0.2, shuffle = True)
#     score = model.evaluate({'G_input' : test_X[:, 0], 'AP_input' : test_X[:, 1 : 750], 'his_input' : test_X[:, 750 :]}, test_Y, 2200)
#     score = pd.Series(score)
#     f_score[i] = score.mean().mean() * 749 * 18
# print(f_score)
# print(f_score.mean())

#predict
i = 553
X = wifi_t.ix[:, i - 3 : i]
X = np.append(wifi_mean.ix[:, i - 144 - 2 : i - 144 + 20], X, axis = 1)
X = np.append(np.tile(wifi_t.ix[:, i - 1].T, (749, 1)), X, axis = 1)
X = np.append(wifi_t.ix[:, 'group'].reshape([749, 1]), X, axis = 1)
res = np.zeros([749, 18])
for k in range(0, 10):
    model.fit({'G_input' : train_X[:, 0], 'AP_input' : train_X[:, 1 : 750], 'his_input' : train_X[:, 750 :]}, train_Y,\
        batch_size=60, nb_epoch=10, shuffle = True, validation_split = 0.2)
    res = res + model.predict({'G_input' : X[:, 0], 'AP_input' : X[:, 1 : 750], 'his_input' : X[:, 750 :]},  batch_size=749)
print(res.shape)
res = res / 10
res[res < 0] = 0

#submit
sub = pd.read_csv('data/sub_template.csv')
res = res.reshape(1, res.shape[0] * res.shape[1])
sub['passengerCount'] = res[0]
sub.to_csv('sub7_keras.csv', index = False)