#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso,RidgeCV,LassoCV,ElasticNet,ElasticNetCV,LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


# In[21]:


pip install pandas-profiling


# In[9]:


from pandas_profiling import ProfileReport


# In[10]:


df = pd.read_csv('Admission_Prediction.csv')


# In[12]:


df.head()


# In[19]:


pf = ProfileReport(df)


# In[20]:


pf.to_widgets()


# In[22]:


df['GRE Score'] = df['GRE Score'].fillna(df['GRE Score'].mean())


# In[23]:


df['TOEFL Score'] = df['TOEFL Score'].fillna(df['TOEFL Score'].mean())


# In[24]:


df['University Rating'] = df['University Rating'].fillna(df['University Rating'].mean())


# In[26]:


df.describe()


# In[25]:


df.isnull().sum()


# In[27]:


df.drop(columns=['Serial No.'],inplace = True)


# In[28]:


df.head(2)


# In[29]:


y=df['Chance of Admit']
x=df.drop(columns='Chance of Admit')


# In[32]:


y.head(2)


# In[33]:


x.head(2)


# In[37]:


scaler = StandardScaler()
#when we do this: mean will be equal to 0 and standard deviation will be equal to 1
#we do this coz o make our algorithm to understand in a better way


# In[38]:


arr = scaler.fit_transform(x)


# In[39]:


arr


# In[40]:


#df1=pd.DataFrame(arr)
#df1.profile_report()


# In[ ]:




