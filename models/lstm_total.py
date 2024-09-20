#!/usr/bin/env tf
# coding: utf-8

# In[1]:


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
import os

import warnings

warnings.filterwarnings("ignore")

# {'DD': 0.001, 'DP2': 0.0004, 'DP1': 0.00035, 'HP1': 0.00005, 'HP2': 0.0001, 'HD': 0.0002}


# In[2]:


df = pd.read_csv(os.path.dirname(__file__) + "/../data/daily_total.csv")
df["DATE"] = pd.to_datetime(df["DATE"])
df.dtypes


# In[3]:


df = df[df["DATE"] > "2022-08-10"]
# fill missing dates with 0
df.index = df["DATE"]
df = df.reindex(pd.date_range(min(df["DATE"]), max(df["DATE"])), fill_value=0).drop(
    "DATE", axis=1
)
df["DATE"] = df.index
# get day of week in a new column
df["DOW"] = df["DATE"].dt.day_name()
df.drop(["DATE", "DOW"], axis=1, inplace=True)
df


# In[4]:


seconds_spent = df["SECONDSSPENT"].values.reshape(-1, 1)
user_count = df["USERCOUNT"].values.reshape(-1, 1)
quantity = df["QUANTITY"].values.reshape(-1, 1)
volume = df["VOLUME"].values.reshape(-1, 1)
weight = df["WEIGHT"].values.reshape(-1, 1)
price = df["PRICE"].values.reshape(-1, 1)
# dow = df['DOW'].values.reshape(-1, 1)
# year = df['DATE'].dt.year.values.reshape(-1, 1)
# month = df['DATE'].dt.month.values.reshape(-1, 1)
# day = df['DATE'].dt.day.values.reshape(-1, 1)

# # one hot encode day of week
# label_encoder = LabelEncoder()
# dow = label_encoder.fit_transform(dow)
# year = label_encoder.fit_transform(year)
# month = label_encoder.fit_transform(month)
# day = label_encoder.fit_transform(day)
# dow = dow.reshape(len(dow), 1)
# year = year.reshape(len(year), 1)
# month = month.reshape(len(month), 1)
# day = day.reshape(len(day), 1)

scaler = MinMaxScaler(feature_range=(0, 1))
# seconds_spent = scaler.fit_transform(seconds_spent)
# user_count = scaler.fit_transform(user_count)
# quantity = scaler.fit_transform(quantity)
# volume = scaler.fit_transform(volume)
# weight = scaler.fit_transform(weight)
# price = scaler.fit_transform(price)

dataset = scaler.fit_transform(hstack((quantity, volume, weight, price, seconds_spent)))


# In[5]:


# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[:train_size, :], dataset[train_size:, :]

train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[6]:


# design network
np.random.seed(1234)
tf.random.set_seed(1234)

tf.keras.utils.set_random_seed(1234)
tf.config.experimental.enable_op_determinism()

optim = optimizers.Adam(lr=0.0005)
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(16, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.add(Dropout(0.1))
model.add(Dense(1, activation="linear"))
model.compile(loss="mae", optimizer=optim)

# fit network
history = model.fit(
    train_X,
    train_y,
    epochs=100,
    batch_size=64,
    validation_data=(test_X, test_y),
    verbose=2,
    shuffle=False,
)

# plot history
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="test")
plt.legend()
plt.show()


# In[7]:


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


# In[8]:


date = df.index[train_size:]

fig, ax = plt.subplots()
fig.set_size_inches(18, 6)
ax.plot(
    date,
    inv_y / 3600,
    linestyle="none",
    marker="s",
    markerfacecolor="cornflowerblue",
    markeredgecolor="black",
    markersize=7,
    label="Hours spent per day",
)
ax.plot(
    date,
    inv_yhat / 3600,
    linestyle="-",
    marker="o",
    markersize=5,
    color="red",
    label="Prediction",
)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%b-%d"))
ax.set_xlabel("Date")
ax.set_ylabel("Hours Spent")
plt.xticks(rotation=75)
ax.legend(loc="upper left")
plt.show()


# In[9]:


# MAE
mae = mean_absolute_error(inv_y, inv_yhat) / 3600
print("MAE: %.3f" % (mae))


# In[16]:


# write all test data with variables
# quantity, volume, weight, price, seconds_spent
# and predictions to a csv file

df_pred = pd.DataFrame(hstack((X, inv_y.reshape(-1, 1), inv_yhat.reshape(-1, 1))))
df_pred.columns = [
    "QUANTITY",
    "VOLUME",
    "WEIGHT",
    "PRICE",
    "SECONDSSPENT",
    "PREDICTION",
]
df_pred["DATE"] = date
df_pred.to_csv(os.path.dirname(__file__) + "/../data/lstm_total_prediction.csv", index=False)


# In[ ]:
