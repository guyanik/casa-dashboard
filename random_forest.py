#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from numpy import hstack
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('data/daily.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df.dtypes


# In[3]:


df = df[df['DATE'] > '2022-08-10']


# In[4]:


teams = df['TEAMNAME'].unique()
df_multi_index = pd.MultiIndex.from_product([df['DATE'].unique(),
                                   teams], names=['DATE', 'TEAMNAME'])
df = df.set_index(['DATE', 'TEAMNAME']).reindex(df_multi_index, fill_value=0).reset_index()
df = df[df['TEAMNAME'] != 'ID9']


# In[5]:


codes = dict(zip(df['TEAMNAME'].unique(), range(1, len(df['TEAMNAME'].unique())+1)))
df['TEAMNAME'] = df['TEAMNAME'].map(codes)
print(codes)
df


# In[6]:


# correlation matrix
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[7]:


# split into train and test sets
cols = ['QUANTITY', 'VOLUME', 'WEIGHT', 'PRICE', 'TEAMNAME', 'USERCOUNT']

train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train, test = df[:train_size], df[train_size:]
train_X, train_y = train[cols], train['SECONDSSPENT']
test_X, test_y = test[cols], test['SECONDSSPENT']


# In[8]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(train_X, train_y)
grid_search.best_params_

best_grid = grid_search.best_estimator_
best_grid


# In[30]:


random_forest_model = RandomForestRegressor(random_state=1234, 
                                            bootstrap=True,
                                            max_depth=100, 
                                            max_features=3,
                                            min_samples_leaf=5, 
                                            min_samples_split=8, 
                                            n_estimators=200,
                                            criterion='mae')
model = BaggingRegressor(base_estimator=random_forest_model, n_estimators=10, random_state=1234)
model.fit(train_X, train_y)


# In[31]:


yhat = model.predict(test_X)


# In[32]:


date = df['DATE'][train_size:]

fig, ax = plt.subplots()
fig.set_size_inches(18, 6)
ax.plot(date, test_y / 3600, 
        linestyle='none', 
        marker='s',
        markerfacecolor='cornflowerblue', 
        markeredgecolor='black',
        markersize=7,
        label='Hours spent per day')
ax.plot(date, yhat / 3600, 
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


# In[33]:


mae = mean_absolute_error(test_y, yhat) / 3600
print('MAE: %.3f' % mae)


# In[ ]:




