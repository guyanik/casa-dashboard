#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import pandas as pd
from numpy import hstack
from numpy import concatenate
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import GaussianNoise
from keras.layers import Masking
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import warnings
warnings.filterwarnings('ignore')

# {'DD': 0.001, 'DP2': 0.0004, 'DP1': 0.00035, 'HP1': 0.00005, 'HP2': 0.0001, 'HD': 0.0002}


# In[28]:


df = pd.read_csv('data/daily_total.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df.dtypes


# In[29]:


df = df[df['DATE'] > '2022-08-10']
# fill missing dates with 0
df.index = df['DATE']
df = df.reindex(pd.date_range(min(df['DATE']), max(df['DATE'])), fill_value=0).drop('DATE', axis=1)
df['SECONDSSPENTNextDay'] = df['SECONDSSPENT'].shift(-1)
df


# In[30]:


seconds_spent_next_day = df['SECONDSSPENTNextDay'].values.reshape(-1, 1)
seconds_spent = df['SECONDSSPENT'].values.reshape(-1, 1)
user_count = df['USERCOUNT'].values.reshape(-1, 1)
quantity = df['QUANTITY'].values.reshape(-1, 1)
volume = df['VOLUME'].values.reshape(-1, 1)
weight = df['WEIGHT'].values.reshape(-1, 1)
price = df['PRICE'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))

dataset = scaler.fit_transform(hstack((quantity, volume, weight, price, seconds_spent_next_day)))


# In[31]:


# def df_to_X_y(dataset, window_size=1):
#     # dataset = df.values
#     X = []
#     y = []
#     for i in range(len(dataset) - window_size):
#         row = [[j] for j in dataset[i: i + window_size, :]]
#         # row = dataset[i: i + window_size]
#         X.append(row)
#         y.append(dataset[i + window_size][-1])
#     return np.array(X), np.array(y)

# split = int(len(dataset) * 0.9)
# window_size = 1

# X, y = df_to_X_y(dataset, window_size)
# print(X.shape, y.shape)
# train_X, train_y = X[:split], y[:split]
# test_X, test_y = X[split:], y[split:]

# train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], train_X.shape[3]))
# test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], test_X.shape[3]))

# # train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
# # test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[32]:


# split into train and test sets
window_size = 1
split = int(len(dataset) * 0.9)

train, test = dataset[:-1][:split], dataset[:-1][split:]

train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[33]:


# design network
np.random.seed(1234)
tf.random.set_seed(1234)

tf.keras.utils.set_random_seed(1234)
tf.config.experimental.enable_op_determinism()

optim = optimizers.Adam(lr=0.0001)
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(16, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(LSTM(8, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer=optim)

# fit network
history = model.fit(train_X, train_y, epochs=70, batch_size=32, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# In[34]:


# make a prediction
yhat = model.predict(test_X)
X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
print(X.shape, yhat.shape)

# invert scaling for forecast
inv_yhat = concatenate((X, yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, -1]

# invert scaling for actual
y = test_y.reshape((len(test_y), 1))
dataset_y = concatenate((X, y), axis=1)
dataset_y = scaler.inverse_transform(dataset_y)
inv_y = dataset_y[:, -1]
# inv_y = scaler.inverse_transform(test_y)
# inv_y = inv_y[:, -1]


# In[35]:


df['DATE'] = df.index
date = df['DATE'][split + window_size:]

fig, ax = plt.subplots()
fig.set_size_inches(18, 6)
ax.plot(date, inv_y / 3600, 
        linestyle='none', 
        marker='s',
        markerfacecolor='cornflowerblue', 
        markeredgecolor='black',
        markersize=7,
        label='Hours spent per day')
ax.plot(date, inv_yhat / 3600, 
        linestyle='-',
        marker='o',
        markersize=5,
        color='red',
        label='Prediction')
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b-%d'))
ax.set_xlabel('Date')
ax.set_ylabel('Hours Spent')
plt.xticks(rotation=75)
ax.legend(loc='upper left')
plt.show()


# In[36]:


# MAE
mae = mean_absolute_error(inv_y[:-1], inv_yhat[1:]) / 3600
print('MAE: %.3f' % (mae))


# In[37]:


last = dataset[-1, :-1].reshape((1, test_X.shape[1], test_X.shape[2]))

# make a prediction
forecast = model.predict(last)
X = last.reshape((last.shape[0], last.shape[2]))
print(X.shape, forecast.shape)

# invert scaling for forecast
inv_forecast = concatenate((X, forecast), axis=1)
inv_forecast = scaler.inverse_transform(inv_forecast)
inv_forecast = inv_forecast[:, -1]


# In[38]:


print('Actual: %.3f' % (df['SECONDSSPENT'][-1] / 3600))
print('Forecast: %.3f' % (inv_forecast / 3600))

