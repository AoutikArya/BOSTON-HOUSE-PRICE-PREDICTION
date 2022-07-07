#!/usr/bin/env python
# coding: utf-8

# In[1]:



from sklearn.datasets import load_boston

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# In[2]:


data_b=load_boston()
x=data_b.data
y=data_b.target


# In[3]:



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=77)

from sklearn.preprocessing import MinMaxScaler


# In[4]:



sc=MinMaxScaler(feature_range=(0,1))
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
y_train=y_train.reshape(-1,1)
y_train=sc.fit_transform(y_train)


# In[5]:



from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)


# In[6]:



predicted_lr=lr.predict(x_test)
predicted_lr=sc.inverse_transform(predicted_lr)



# In[7]:



from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[8]:


mae=mean_absolute_error(y_test,predicted_lr)
mse=mean_squared_error(y_test,predicted_lr)
rmse=math.sqrt(mse)


# In[ ]:


print(mae)
print(mse)

