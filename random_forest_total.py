#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numpy as np
import pandas as pd
from numpy import hstack
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pprint import pprint

import warnings
warnings.filterwarnings('ignore')


# In[66]:


df = pd.read_csv('data/daily_total.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df.dtypes


# In[67]:


df = df[df['DATE'] >= '2022-08-10']


# In[68]:


# split into train and test sets
cols = ['QUANTITY', 'VOLUME', 'WEIGHT', 'PRICE', 'USERCOUNT']

train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train, test = df[:train_size], df[train_size:len(df)]
train_X, train_y = train[cols], train['SECONDSSPENT']
test_X, test_y = test[cols], test['SECONDSSPENT']


# In[69]:


random_forest_model = RandomForestRegressor(random_state=1234)
print('Parameters currently in use:\n')
pprint(random_forest_model.get_params())


# In[70]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'criterion': ['mae']}
pprint(random_grid)


# In[71]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor(random_state=1234)
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=random_forest_model, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=1234, n_jobs=-1)
# Fit the random search model
rf_random.fit(train_X, train_y)


# In[72]:


rf_random.best_params_


# In[79]:


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f}'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(train_X, train_y)
base_accuracy = evaluate(base_model, test_X, test_y)


# In[80]:


best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, test_X, test_y)


# In[81]:


print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))


# In[82]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000],
    'criterion': ['mae']
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)


# In[83]:


# Fit the grid search to the data
grid_search.fit(train_X, train_y)
grid_search.best_params_

best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, test_X, test_y)


# In[84]:


best_grid


# In[57]:


print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))


# In[85]:


# random_forest_model = RandomForestRegressor(random_state=1234, 
#                                             bootstrap=True, 
#                                             max_depth=50, 
#                                             max_features='auto',
#                                             min_samples_leaf=4, 
#                                             min_samples_split=2, 
#                                             n_estimators=1600,
#                                             criterion='mae')

random_forest_model = RandomForestRegressor(random_state=1234, 
                                            max_depth=100, 
                                            max_features=3,
                                            min_samples_leaf=5, 
                                            min_samples_split=10, 
                                            n_estimators=200,
                                            criterion='mae')


# In[86]:


model = BaggingRegressor(base_estimator=random_forest_model, random_state=1234)
model.fit(train_X, train_y)


# In[87]:


yhat = model.predict(test_X)


# In[88]:


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


# In[89]:


mae = mean_absolute_error(test_y, yhat) / 3600
print('MAE: %.3f' % mae)


# In[ ]:




