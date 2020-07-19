#!/usr/bin/env python
# coding: utf-8

# ## 第一部分数据预处理
# 流程如下：
# ####  1. 导入必要的包并读取数据 
# ####  2. 对两个表分别进行预处理，预处理流程如下：
# ##### 对c1表：去除卡号相同的重复值、采用填充法进行缺失值处理、异常值检测
# #####  对c2表：去除重复值、缺失值检测、异常值处理（如销售金额为负数、购买数量为负数等）
# #### 3. 合并两个表便于后续处理，连接方式为外连接

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on 2020-07-19

@author: 豆奶
"""
# 导入必要的包
import pandas as pd


# ## 对c1表的预处理

# In[2]:


data1 = pd.read_excel("../data/cumcm2018c1.xlsx")
data1.head()


# In[3]:


# 去除卡号重复值
print("去除重复值前的数据量", data1.shape)   
data1.drop_duplicates(subset=['kh'],keep='first',inplace=True)
print("去除重复值后的数据量", data1.shape)  


# In[4]:


# 缺失值处理
print('处理前缺失值数目是：\n', data1.isnull().sum())  # 可以看到缺失值较多，采取填充法进行缺失值处理
data1 = data1.fillna(axis=0,method='ffill')   # 填充法
print('处理后缺失值数目是：\n', data1.isnull().sum())


# ## 对c2表的预处理

# In[6]:


data2 = pd.read_csv("../data/cumcm2018c2.csv")
data2.head()


# In[7]:


# 去除重复值
print("去除重复值前的数据量", data2.shape)  
data2.drop_duplicates(subset=None,keep='first',inplace=True)
print("去除重复值后的数据量", data2.shape)  


# In[9]:


# 缺失值检测
print('处理前缺失值数目是：\n', data2.isnull().sum())  
# 可以看到缺失值部分为：会员卡号，积分，柜组编码，柜组名，因此不需要进行填充


# In[10]:


# 异常值检测
print(data2.min(), '\n')
print(data2.max())


# In[11]:


# 异常值处理，对于异常值的阈值，可与商场负责人进行确认，这里暂时凭个人感觉进行确定
print("去除异常值前的数据量", data2.shape)  

data2 = data2.drop(data2[(data2.je< 0)].index)  # 去除销售金额为负数的数据
data2 = data2.drop(data2[(data2.je>5000)].index) # 去除销售金额过大的数据，这里暂时定为5000，具体可与商场进行商讨

data2 = data2.drop(data2[(data2.sj< 0.5)].index)  # 去除售价为负数的数据
data2 = data2.drop(data2[(data2.sj>5000)].index) # 去除售价过大的数据，这里暂时定为5000

data2 = data2.drop(data2[(data2.sl< 0)].index)  # 去除购买数量为负数的数据
data2 = data2.drop(data2[(data2.sl> 1000)].index)  # 去除购买数量过大的数据

data2 = data2.drop(data2[(data2.jf< 0)].index)  # 去除积分为负数的数据

# 处理后的结果查看
print("去除异常值后的数据量", data2.shape)  
print(data2.min(), '\n')
print(data2.max())


# ## 合并两张表

# In[16]:


result = pd.merge(data1, data2, on='kh', how="outer")


# In[17]:


print(result.shape) 
result.head()


# In[18]:


result.to_csv('../data/task1.csv',  encoding='utf-8')


# In[ ]:





# In[ ]:




