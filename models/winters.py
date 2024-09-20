#!/usr/bin/env python
# coding: utf-8

# In[323]:


import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os

import warnings
warnings.filterwarnings("ignore")


# In[324]:


df = pd.read_csv(os.path.dirname(__file__) + '/../data/daily_total.csv')
df.index = [datetime.strptime(x, '%Y-%m-%d') for x in df['DATE']]
df = df[df.index > '2022-08-10']
df = df.reindex(pd.date_range(min(df['DATE']), max(df['DATE'])), fill_value=0).drop('DATE', axis=1)
column = 'SECONDSSPENT'
df


# In[325]:


df.resample("YS").sum().round(2).corr().style.background_gradient(cmap='coolwarm')


# In[326]:


# correlation matrix
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[327]:


fig, ax = plt.subplots()
fig.set_size_inches(18, 6)
ax.plot(df.index, df[column] / 3600, linestyle='none', marker='s',
        markerfacecolor='cornflowerblue', 
        markeredgecolor='black',
        markersize=7,
        label='Hours spent per day')
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
ax.set_xlabel('Date')
ax.set_ylabel('Hours Spent')
plt.xticks(rotation=60)
ax.legend(loc='upper left')


# In[328]:


# boxplot of hours spent per day
fig, ax = plt.subplots()
fig.set_size_inches(6, 6)
ax.boxplot(df[column] / 3600)
ax.set_ylabel('Hours Spent')
ax.set_title('Boxplot of Hours Spent per Day')
plt.show()


# In[329]:


df[column].sort_values(ascending=False).head(10)


# In[330]:


# # remove outliers
# df = df[df['SECONDSSPENT'] / 3600 < 2000]

# fig, ax = plt.subplots()
# fig.set_size_inches(18, 6)
# ax.plot(df.index, df['SECONDSSPENT'] / 3600, linestyle='none', marker='s',
#         markerfacecolor='cornflowerblue', 
#         markeredgecolor='black',
#         markersize=7,
#         label='Hours spent per day')
# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
# ax.set_xlabel('Date')
# ax.set_ylabel('Hours Spent')
# plt.xticks(rotation=60)
# ax.legend(loc='upper left')


# In[331]:


corr_list = []

for i in range(1, len(df) - 1):
    corr_list.append(np.corrcoef(df[column][i:], df[column][: -i])[0, 1])

# index and correlation coefficient dataframe
corr_df = pd.DataFrame({'Lag': list(range(1, len(df) - 1)), 'Correlation': corr_list})
corr_df = corr_df.sort_values(by='Correlation', ascending=False)
print(corr_df.head(10).to_string(index=False))


# In[332]:


def find_period(signal, corr_threshold=0.2):
    acf = np.correlate(signal, signal, 'full')[-len(signal):]
    inflection = np.diff(np.sign(np.diff(acf)))
    peaks = (inflection < 0).nonzero()[0] + 1
    if len(peaks) == 0:
        print('No seasonality found')
        return 1, 0
    period = peaks[acf[peaks].argmax()]
    corr = np.corrcoef(df[column][period:], df[column][: -period])[0, 1]
    if corr < corr_threshold or period == 1:
        print('No seasonality found')
        return 1, corr
    return period, corr

period, corr = find_period(df[column])
print('Period:', period, '\nCorrelation:', corr)


# In[333]:


# log transformation
df_log = df.copy()
df_log[column] = df_log[column].replace(0, 3600)
# df_log[column] = df_log[column].replace(0, np.nan).interpolate(method='cubicspline')
# df_log[column] = np.log(df_log[column]).interpolate(method='cubicspline')


# In[334]:


df_log[column].isna().sum()


# In[335]:


# deseaonalize
df_log['Seasonal'] = seasonal_decompose(df_log[column], model='additive', period=period).seasonal
df_log['DS'] = seasonal_decompose(df_log[column], model='additive', period=period).trend

# linear regression with deseasonalized SECONDSSPENT
reg_df = df_log[df_log['DS'].notna()]
endog = np.array(reg_df['DS'])
exog = sm.add_constant(np.array(reg_df.index.dayofyear))
model = sm.OLS(endog, exog)
results = model.fit()
# print(results.summary())

initial_level = results.params[0]
initial_trend = results.params[1]
df_log['DS'] = initial_level + initial_trend * np.array(df_log.index.dayofyear)
df_log['Seasonal Factor'] = df_log[column] / df_log['DS']

initial_seasonal = []
if period == 1:
    initial_seasonal = [1]
else:
    for i in range(period):
        initial_seasonal.append(df_log.iloc[i::period].mean()['Seasonal Factor'])

print(initial_level)
print(initial_trend)
print(initial_seasonal)


# In[336]:


seasonal_periods = period
alpha = 0.5
beta = 0.01
gamma = 0.01

winters_model = ExponentialSmoothing(
    df_log[column],
    trend="add",
    seasonal="mul",
    seasonal_periods=seasonal_periods,
    initialization_method="known",
    initial_level=initial_level,
    initial_trend=initial_trend,
    initial_seasonal=initial_seasonal
).fit(smoothing_level=alpha,
      smoothing_trend=beta,
      smoothing_seasonal=gamma,
      )

last_day = df_log.index[-1]

num_periods = period
df_log["FORECAST"] = winters_model.fittedvalues
forecast = pd.DataFrame(
    {'FORECAST': list(winters_model.forecast(num_periods))},
    index=pd.date_range(start=last_day + pd.DateOffset(days=1), periods=num_periods, freq='D')
)

df_forecast = df_log.append(forecast)
# df_forecast[[column,'FORECAST']] = np.exp(df_forecast[[column,'FORECAST']])
df_forecast.tail(period*2)


# In[337]:


df_forecast['DATE'] = df_forecast.index
df_forecast[['DATE', 'SECONDSSPENT', 'FORECAST']].to_csv(os.path.dirname(__file__) + '/../data/hw_forecast.csv', index=False)


# In[341]:


last_day = df.index[-1]
# df_plot = df_forecast[df_forecast.index >= last_day - pd.to_timedelta("60day")]
# df_plot = df_forecast[df_forecast.index <= last_day]
# last sixty observations
df_plot = df_forecast[-12:]

fig, ax = plt.subplots()
fig.set_size_inches(18, 6)
ax.plot(df_plot.index, df_plot[column] / 3600, 
        linestyle='none', 
        marker='s',
        markerfacecolor='cornflowerblue', 
        markeredgecolor='black',
        markersize=7,
        label='Hours spent per day')
ax.plot(df_plot.index, df_plot['FORECAST'] / 3600, 
        linestyle='-',
        marker='o',
        markersize=5,
        color='red',
        label='Forecast')
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b-%d'))
ax.set_xlabel('Date')
ax.set_ylabel('Hours Spent')
plt.xticks(rotation=75)
ax.legend(loc='upper left')
plt.show()


# In[342]:


# mean absolute error
mae = np.mean(np.abs(df_plot[column] - df_plot['FORECAST'])) / 3600
print('MAE: {:.3f}'.format(mae))

# root mean squared error
mse = np.sqrt(np.mean(((df_plot[column] - df_plot['FORECAST'])) ** 2)) / 3600
print('RMSE: {:.3f}'.format(mse))


# In[319]:


import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def _get_period(data: pd.Series, corr_threshold: float = 0.2):
    acf = np.correlate(data, data, 'full')[-len(data):]
    inflection = np.diff(np.sign(np.diff(acf)))
    peaks = (inflection < 0).nonzero()[0] + 1
    if len(peaks) == 0:
        print('No seasonality found')
        return 1, 0
    period = peaks[acf[peaks].argmax()]
    corr = np.corrcoef(data[period:], data[: -period])[0, 1]
    if corr < corr_threshold or period == 1:
        print('No seasonality found')
        return 1, corr
    return period, corr


def _get_initial_params(df: pd.DataFrame, column: str, log: bool = True):
    period, corr = _get_period(df[column])

    df_log = df.copy()
    if log:
        df_log[column] = df_log[column].replace(0, np.nan).interpolate(method='cubicspline')
        df_log[column] = np.log(df_log[column])
    else:
        df_log[column] = df_log[column].replace(0, 3600)

    # deseaonalize
    df_log['Seasonal'] = seasonal_decompose(df_log[column], model='additive', period=period).seasonal
    df_log['DS'] = seasonal_decompose(df_log[column], model='additive', period=period).trend

    # linear regression with deseasonalized SECONDSSPENT
    reg_df = df_log[df_log['DS'].notna()]
    endog = np.array(reg_df['DS'])
    exog = sm.add_constant(np.array(reg_df.index.dayofyear))
    model = sm.OLS(endog, exog)
    results = model.fit()
    # print(results.summary())

    initial_level = results.params[0]
    initial_trend = results.params[1]
    df_log['DS'] = initial_level + initial_trend * np.array(df_log.index.dayofyear)
    df_log['Seasonal Factor'] = df_log[column] / df_log['DS']

    initial_seasonal = []
    if period == 1:
        initial_seasonal = [1]
    else:
        for i in range(period):
            initial_seasonal.append(df_log.iloc[i::period].mean()['Seasonal Factor'])
    
    return period, initial_level, initial_trend, initial_seasonal


def forecast(df: pd.DataFrame, 
            column: str, 
            log: bool = False,
            alpha: float = 0.5, 
            beta: float =  0.01, 
            gamma: float = 0.01):
    
    period, initial_level, initial_trend, initial_seasonal = _get_initial_params(df, column, log)
    print('Period: {}'.format(period))
    print('Initial Level: {}'.format(initial_level))
    print('Initial Trend: {}'.format(initial_trend))
    print('Initial Seasonal: {}'.format(initial_seasonal))
    
    df_log = df.copy()
    if log:
        df_log[column] = df_log[column].replace(0, np.nan).interpolate(method='cubicspline')
        df_log[column] = np.log(df_log[column])
    else:
        df_log[column] = df_log[column].replace(0, 3600)
    
    if period != 1:
        winters_model = ExponentialSmoothing(
        df_log[column],
        trend="add",
        seasonal="mul",
        seasonal_periods=period,
        initialization_method="known",
        initial_level=initial_level,
        initial_trend=initial_trend,
        initial_seasonal=initial_seasonal
    ).fit(smoothing_level=alpha,
        smoothing_trend=beta,
        smoothing_seasonal=gamma,
        )
    else:
        winters_model = ExponentialSmoothing(
        df_log[column],
        trend="add",
        seasonal=None,
        initialization_method="known",
        initial_level=initial_level,
        initial_trend=initial_trend
    ).fit(smoothing_level=alpha,
        smoothing_trend=beta
        )

    last_day = df_log.index[-1]
    num_periods = period
    df_log["FORECAST"] = winters_model.fittedvalues
    forecast = pd.DataFrame(
        {'FORECAST': list(winters_model.forecast(num_periods))},
        index=pd.date_range(start=last_day + pd.DateOffset(days=1), periods=num_periods, freq='D')
    )

    df_forecast = df_log.append(forecast)
    if log:
        df_forecast[[column,'FORECAST']] = np.exp(df_forecast[[column,'FORECAST']])
    print(df_forecast.tail(period*2))

    return df_forecast


def plot_forecast(df: pd.DataFrame, column: str, title: str = ''):
    df_forecast = forecast(df, column)
    # df_plot = df_forecast[df_forecast.index >= '2022-08-10']
    df_plot = df_forecast[-66:]

    test_size = int(len(df_plot) * 0.2)
    test = df_plot[-test_size:]
    mae = np.mean(np.abs(test[column] - test['FORECAST'])) / 3600
    print('MAE: {:.3f}'.format(mae))

    fig, ax = plt.subplots()
    fig.set_size_inches(18, 6)
    ax.plot(df_plot.index, df_plot[column] / 3600, 
            linestyle='none', 
            marker='s',
            markerfacecolor='cornflowerblue', 
            markeredgecolor='black',
            markersize=7,
            label='Hours spent per day')
    ax.plot(df_plot.index, df_plot['FORECAST'] / 3600, 
            linestyle='-',
            marker='o',
            markersize=5,
            color='red',
            label='Forecast')
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b-%d'))
    ax.set_xlabel('Date')
    ax.set_ylabel('Hours Spent')
    plt.xticks(rotation=75)
    ax.legend(loc='upper left')
    plt.show()



# In[321]:


df.shape


# In[320]:


df = pd.read_csv(os.path.dirname(__file__) + '/../data/daily.csv')
df.index = pd.to_datetime(df['DATE'])
df = df[df.index > '2022-08-10']
df_multi_index = pd.MultiIndex.from_product([df['DATE'].unique(), df['TEAMNAME'].unique()], names=['DATE', 'TEAMNAME'])
df = df.set_index(['DATE', 'TEAMNAME']).reindex(df_multi_index, fill_value=0).reset_index()
df.index = pd.to_datetime(df['DATE'])
column = 'SECONDSSPENT'
print(0 in df[column].to_list())
for team in df['TEAMNAME'].unique():
    df_team = df[df['TEAMNAME'] == team]
    if len(df_team) > 1:
        print(team)
        plot_forecast(df_team, column, team)


# In[ ]:




