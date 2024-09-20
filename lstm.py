#!/usr/bin/env python
# coding: utf-8

# In[76]:


import numpy as np
import pandas as pd
from numpy import hstack
from numpy import concatenate
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers
from keras.layers import Masking
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import warnings
warnings.filterwarnings('ignore')


# In[77]:


df = pd.read_csv('data/daily.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df.dtypes


# In[78]:


df = df[(df['DATE'] > '2022-08-10')]
# df = df[(df['DATE'] != '2020-07-22')]


# In[79]:


df.columns


# In[80]:


df = df[df['TEAMNAME'] != 'ID9']

df.index = pd.DatetimeIndex(df['DATE'])
df = df.drop(columns=['DATE'])
s = pd.DataFrame(index=pd.date_range(df.index.min(), df.index.max(), freq='D').difference(df.index), columns=df.columns)
df = pd.concat([df,s]).fillna(0).sort_index()
df['DATE'] = df.index
df = df.reset_index(drop=True)
df['TEAMNAME'] = df['TEAMNAME'].replace(0, 'DP1')

df_multi_index = pd.MultiIndex.from_product([df['DATE'].unique(), df['TEAMNAME'].unique()], names=['DATE', 'TEAMNAME'])
df = df.set_index(['DATE', 'TEAMNAME']).reindex(df_multi_index, fill_value=0).reset_index()
df

# write to csv
# df.to_csv('data/try.csv')

# df_dp = df.loc[df['TEAMNAME'].isin(['DP1', 'DP2']), :]
# df_rest = df.loc[df['TEAMNAME'].isin(['DD', 'HD', 'HP1', 'HP2']), :].groupby('DATE').sum().reset_index()
# df_rest['TEAMNAME'] = 'REST'
# df = pd.concat([df_dp, df_rest])
# df = df.sort_values(by=['DATE', 'TEAMNAME'])


# In[81]:


df['MONTH'] = df['DATE'].dt.month
df['DAY'] = df['DATE'].dt.day


# In[82]:


codes = dict(zip(df['TEAMNAME'].unique(), range(0, len(df['TEAMNAME'].unique()))))
codes


# In[83]:


# codes = {'DD': 0.001, 'DP2': 0.0004, 'DP1': 0.00035, 'HP1': 0.00005, 'HP2': 0.0001, 'HD': 0.0002}
# df['TEAMNAME'] = 1 / df['TEAMNAME'].map(codes)
# print(codes)
# df


# In[84]:


seconds_spent = df['SECONDSSPENT'].values.reshape(-1, 1)
user_count = df['USERCOUNT'].values.reshape(-1, 1)
quantity = df['QUANTITY'].values.reshape(-1, 1)
volume = df['VOLUME'].values.reshape(-1, 1)
weight = df['WEIGHT'].values.reshape(-1, 1)
price = df['PRICE'].values.reshape(-1, 1)
team_name = df['TEAMNAME'].values.reshape(-1, 1)
date = df['DATE'].values.reshape(-1, 1)
month = df['MONTH'].values.reshape(-1, 1)
day = df['DAY'].values.reshape(-1, 1)

# one hot encode TEAMNAME
onehot_encoder = OneHotEncoder(sparse_output=False)
team_name = onehot_encoder.fit_transform(df[['TEAMNAME']])
date = onehot_encoder.fit_transform(date)
month = onehot_encoder.fit_transform(month)
day = onehot_encoder.fit_transform(day)

scaler = MinMaxScaler(feature_range=(0, 1))
# seconds_spent = scaler.fit_transform(seconds_spent)
# user_count = scaler.fit_transform(user_count)
# quantity = scaler.fit_transform(quantity)
# volume = scaler.fit_transform(volume)
# weight = scaler.fit_transform(weight)
# price = scaler.fit_transform(price)

dataset = scaler.fit_transform(hstack((quantity, volume, weight, price, team_name, seconds_spent)))
team_name.shape


# In[85]:


# split into train and test sets
train_size = int(len(dataset) * 0.9)
test_size = len(dataset) - train_size

train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[86]:


# design network
np.random.seed(1234)
tf.random.set_seed(1234)

tf.keras.utils.set_random_seed(1234)
tf.config.experimental.enable_op_determinism()

optim = optimizers.Adam(lr=0.0001, decay=1e-6)

model = Sequential()
model.add(Masking(mask_value=0., input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.4))
model.add(Dense(1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mae', optimizer=optim)

# fit network
history = model.fit(train_X, train_y, epochs=300, batch_size=64, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# In[87]:


# make a prediction
yhat = model.predict(test_X)
X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for prediction
dataset_yhat = concatenate((X, yhat), axis=1)
dataset_yhat = scaler.inverse_transform(dataset_yhat)
inv_yhat = dataset_yhat[:, -1]

# invert scaling for actual
y = test_y.reshape((len(test_y), 1))
dataset_y = concatenate((X, y), axis=1)
dataset_y = scaler.inverse_transform(dataset_y)
inv_y = dataset_y[:, -1]


# In[88]:


date = df['DATE'][train_size:]
num_teams = df['TEAMNAME'].nunique()

fig, ax = plt.subplots()
fig.set_size_inches(18, 6)
ax.plot(date, inv_y / 3600, 
        linestyle='none', 
        marker='s',
        markerfacecolor='cornflowerblue', 
        markeredgecolor='black',
        markersize=7,
        label='Hours spent per day')
# ax.plot(date, inv_yhat / 3600, 
#         linestyle='',
#         marker='o',
#         markersize=5,
#         color='red',
#         label='Prediction')
for i in range(num_teams):
        ax.plot(date.unique(), inv_yhat[i::num_teams] / 3600, 
                linestyle='-',
                marker='o',
                markersize=5,
                # get color from color cycle
                color=ax._get_lines.get_next_color(),
                label=list(codes.keys())[i])
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b-%d'))
ax.set_xlabel('Date')
ax.set_ylabel('Hours Spent')
plt.xticks(rotation=75)
ax.legend(loc='upper left')
plt.show()


# In[89]:


# MAE
mae = mean_absolute_error(inv_y, inv_yhat) / 3600
print('MAE: %.3f' % (mae))


# In[90]:


# # design network
# np.random.seed(1234)
# tf.random.set_seed(1234)

# tf.keras.utils.set_random_seed(1234)
# tf.config.experimental.enable_op_determinism()

# optim = optimizers.Adam(lr=0.0005, beta_1=0.99, beta_2=0.999, amsgrad=False)
# model = Sequential()
# model.add(LSTM(4, input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(Dense(1))
# # model.add(Dropout(0.1))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='mae', optimizer=optim)

# # fit network
# history = model.fit(train_X, train_y, epochs=100, batch_size=16, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# # plot history
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show()


# In[91]:


# write all test data with
# team names as columns
# and predictions to a csv file

df['HOURSSPENT'] = df['SECONDSSPENT'] / 3600
df_pred = df.pivot(index="DATE", columns='TEAMNAME', values='HOURSSPENT')[-len(date.unique()):].reset_index()
# df_pred["DATE"] = date.unique()
# df_pred.columns = ["DATE"] + list(codes.keys())
for i in range(num_teams):
    df_pred[list(codes.keys())[i] + "_pred"] = inv_yhat[i::num_teams] / 3600
df_pred.to_csv('data/lstm_prediction.csv', index=False)


# In[ ]:




