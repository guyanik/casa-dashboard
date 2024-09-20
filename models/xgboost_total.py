#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('data/daily_total.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df.dtypes


# In[3]:


df = df[df['DATE'] > '2022-08-10']


# In[4]:


codes = dict(zip(df['DATE'].unique(), range(1, len(df['DATE'].unique())+1)))
df['DATE'] = df['DATE'].map(codes)
print(codes)
df


# In[5]:


# correlation matrix
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[6]:


# split into train and test sets
cols = ['QUANTITY', 'VOLUME', 'WEIGHT', 'PRICE']

train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train, test = df[:train_size], df[train_size:]
train_X, train_y = train[cols], train['SECONDSSPENT']
test_X, test_y = test[cols], test['SECONDSSPENT']


# In[7]:


dtrain_reg = xgb.DMatrix(train_X, train_y, enable_categorical=True)
dtest_reg = xgb.DMatrix(test_X, test_y, enable_categorical=True)


# In[8]:


params = {"objective": "reg:squarederror", 
          "tree_method": "hist", 
          "learning_rate": 0.001, 
          "max_depth": 1,
          "subsample": 0.8,
          "colsample_bytree": 0.8,
          "seed": 1234, 
          "eval_metric": "mae"}
evals = [(dtrain_reg, "train"), (dtest_reg, "test")]
n = 10000

model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n,
   evals=evals,
   verbose_eval=50,
   early_stopping_rounds=50
)


# In[9]:


xgb.plot_importance(model)
plt.figure(figsize = (16, 12))
plt.show()


# In[10]:


yhat = model.predict(dtest_reg)
date = df.index[train_size:]

fig, ax = plt.subplots()
fig.set_size_inches(18, 6)
ax.plot(date, test_y / 3600, 
        linestyle='none', 
        marker='s',
        markerfacecolor='cornflowerblue', 
        markeredgecolor='black',
        markersize=7,
        label='Test')
ax.plot(date, yhat / 3600, 
        linestyle='-',
        marker='o',
        markersize=3,
        color='red',
        label='Prediction')
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b-%d'))
ax.set_xlabel('Date')
ax.set_ylabel('Hours spent')
plt.xticks(rotation=75)
ax.legend(loc='upper right')
plt.show()

mae = mean_absolute_error(test_y, yhat) / 3600
print('MAE: %.3f' % mae)


# In[ ]:




