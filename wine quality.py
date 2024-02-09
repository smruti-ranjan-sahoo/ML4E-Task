#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[13]:


wine=pd.read_csv(r"C:\Users\smrut\OneDrive\Desktop\winequality-white.csv", delimiter=';')


# In[14]:


wine


# In[48]:


wine.describe()


# In[15]:


wine.isnull().sum()


# In[16]:


for i in wine.columns:
    print(i,' ',wine[i].unique())


# In[17]:


sns.countplot(x= wine['quality'])
plt.xlabel('Wine quality')
plt.ylabel('Frequancy')
plt.title('Wine quality classes')


# In[32]:


plt.figure(figsize=(10,10))
corr = wine.corr()
sns.heatmap(corr ,annot=True);


# In[18]:


wine.columns


# In[34]:


ip=wine[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]
op=wine['quality']


# In[35]:


print(ip,'&',op)


# In[36]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(ip,op,train_size=0.8)


# In[37]:


print(x_train,x_test)


# In[38]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[39]:


print(x_test,x_train)


# In[40]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)


# In[41]:


y_pred=lr.predict(x_test)
y_pred


# In[ ]:




