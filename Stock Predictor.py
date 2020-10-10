#!/usr/bin/env python
# coding: utf-8

# In[103]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import mean_squared_error
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA


# In[104]:


supplies = pd.read_csv('Supplies.csv', parse_dates=[0], index_col=[0])


# In[105]:


supplies.head()


# In[106]:


f_supplies = supplies.iloc[:, [0]]


# In[107]:


f_supplies.head()


# In[108]:


type(f_supplies)


# In[109]:


series_value = f_supplies.values


# In[110]:


type(series_value)


# In[111]:


f_supplies.size


# In[112]:


f_supplies.tail()


# In[113]:


f_supplies.describe()


# In[114]:


f_supplies.plot()


# In[121]:


f_supplies_mean = f_supplies.rolling(window = 50).mean()


# In[122]:


f_supplies.plot()
f_supplies_mean.plot()


# In[123]:


value = pd.DataFrame(series_value)


# In[124]:


supplies_df = pd.concat([value, value.shift(1)], axis=1)


# In[125]:


supplies_df.head()


# In[126]:


supplies_df.columns = ['Actual_usage', 'Forecast_usage']


# In[127]:


supplies_df.head()


# In[128]:


supplies_test = supplies_df[1:]


# In[129]:


supplies_test.head()


# In[130]:


supplies_error = mean_squared_error(supplies_test.Actual_usage, supplies_test.Forecast_usage)


# In[131]:


supplies_error


# In[132]:


np.sqrt(supplies_error)


# In[133]:


# plot_acf is to identify parameter Q
# ARIMA(p, d, q)

plot_acf(f_supplies)


# In[134]:


# to identify the value of p

plot_pacf(f_supplies)


# In[135]:


f_supplies.size


# In[136]:


# p=2,3 d=0 q=3,4

supplies_train = f_supplies[0:330]
supplies_test = f_supplies[330:365]


# In[144]:


supplies_model = ARIMA(supplies_train, order=(4, 0, 3))


# In[145]:


supplies_model_fit = supplies_model.fit()


# In[146]:


supplies_model_fit.aic


# In[147]:


supplies_forecast = supplies_model_fit.forecast(steps = 35)[0]


# In[148]:


supplies_forecast


# In[149]:


supplies_test


# In[150]:


np.sqrt(mean_squared_error(supplies_test, supplies_forecast))


# In[ ]:




