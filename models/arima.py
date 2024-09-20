#!/usr/bin/env python
# coding: utf-8

# In[70]:


import numpy as np
import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


# In[71]:


df = pd.read_csv(os.path.dirname(__file__) + "/../daily_total.csv")
df['DATE'] = pd.to_datetime(df['DATE'])
df = df[df['DATE'] > '2022-08-10']
# fill missing dates with 0
df.index = df['DATE']
df = df.reindex(pd.date_range(min(df['DATE']), max(df['DATE'])), fill_value=0).drop('DATE', axis=1)
df


# In[72]:


periods = 7

model = pm.auto_arima(df['SECONDSSPENT'], 
                      m=periods, 
                      seasonal=True, 
                      start_p=0, 
                      start_q=0, 
                      max_order=4, 
                      test='adf', 
                      trace=True, 
                      error_action='ignore', 
                      suppress_warnings=True, 
                      stepwise=True)


# In[73]:


# split into train and test sets
# train_size = int(len(df) * 0.8)
# test_size = len(df) - train_size
train, test = df[:-periods], df[-periods:]
print(train.shape, test.shape)


# In[74]:


model.fit(train['SECONDSSPENT'])
model.summary()


# In[75]:


# add 7 days to df
last_day = df.index[-1]
forecast = model.predict(n_periods=periods + 7, return_conf_int=True)
forecast_range = pd.date_range(start=last_day - pd.DateOffset(days=periods - 1), periods=periods + 7, freq='D')
forecast_df = pd.DataFrame(forecast[0], index=forecast_range, columns=['FORECAST'])
df_forecast = pd.concat([df['SECONDSSPENT'], forecast_df], axis=1)
df_forecast.tail(14)


# In[76]:


# df_forecast['DATE'] = df_forecast.index
# df_forecast.to_csv(os.path.dirname(__file__) + "/../arima_forecast.csv", index=False)


# In[82]:


df_plot = df_forecast[-(periods + 5):]
date = df_plot.index

fig, ax = plt.subplots()
fig.set_size_inches(18, 6)
ax.plot(date, df_plot['SECONDSSPENT'] / 3600, 
        linestyle='none', 
        marker='s',
        markerfacecolor='cornflowerblue', 
        markeredgecolor='black',
        markersize=7,
        label='Hours spent per day')
ax.plot(date, df_plot['FORECAST'] / 3600, 
        linestyle='-',
        marker='o',
        markersize=5,
        color='red',
        label='Forecast')
# set min and max of y-axis
ax.set_ylim(bottom=0)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b-%d'))
ax.set_xlabel('Date')
ax.set_ylabel('Hours Spent')
plt.xticks(rotation=75)
ax.legend(loc='upper left')
plt.show()


# In[81]:


mae = np.mean(np.abs(df_plot['SECONDSSPENT'] - df_plot['FORECAST'])) / 3600
print('MAE: %.3f hours' % mae)


# In[79]:


# fit ARIMA(1,0,0)(2,0,0)[7]

