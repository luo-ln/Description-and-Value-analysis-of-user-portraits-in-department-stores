#!/usr/bin/env python
# coding: utf-8

# ## 第三部分：会员用户画像
# 流程如下：
# ####  1 构建会员用户基本特征标签（性别xb，年龄age，会员积分jf）
# ####  2 构建会员用户业务特征标签（入会时长rhsc--新、老会员，消费水平--高中低等消费）
# ####  3 构建会员用户偏好特征标签（购物时间的偏好，购物季节偏好，购物类型的偏好）
# ####  4 建立用户画像
# 

# ### 3.0 导入必要的库和数据

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on 2020-07-19

@author: 豆奶
"""
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']


# In[2]:


data_task1 = pd.read_csv(r"../data/task1.csv")
member = data_task1[['kh', 'xb', 'csrq', 'jf', 'djsj', 'je', 'dtime', 'spmc']]

print(member.shape)
print('处理前缺失值数目是：\n', member.isnull().sum())
member.dropna(inplace=True)
print('\n处理后缺失值数目是：\n', member.isnull().sum())
print(member.shape)

member.loc[:,'dtime'] = pd.to_datetime(member.loc[:,'dtime'],infer_datetime_format=True) # 将年份转为datatime格式 
member.loc[:,'csrq'] = pd.to_datetime(member.loc[:,'csrq'],infer_datetime_format=True, errors="coerce")   # 将年份转为datatime格式 
member.loc[:,'djsj'] = pd.to_datetime(member.loc[:,'djsj'],infer_datetime_format=True) # 将年份转为datatime格式 


# ### 年龄、入会时长 、购物时间偏好、购物季节偏好

# In[3]:


now_year = datetime.datetime.today().year      #当前的年份
member['age'] = now_year-member.csrq.dt.year   # 年龄
member['rhsc'] = now_year-member.djsj.dt.year  # 入会时长

member = member.drop(member[(member.age > 100)].index)   #  删除异常值与缺失值
member.dropna(inplace=True)


# In[4]:


def get_gwsjp( x ):   # 购物时间偏好
    time_6 = datetime.datetime.strptime('6:00','%H:%M').time()
    time_12 = datetime.datetime.strptime('12:00','%H:%M').time()
    time_18 = datetime.datetime.strptime('18:00','%H:%M').time()
    xtime = x.timetz()
    #xtime = x.dtime.dt.time
    if xtime<time_6:
        return '凌晨'
    elif xtime<time_12:
        return '上午'
    elif xtime<time_18:
        return '下午'
    else:
        return '晚上'
    
member['gwsjp'] = member['dtime'].map(get_gwsjp)


# In[5]:


def get_gwjjp( x ):   # 购物季节偏好
    xmonth = x.month
    #xtime = x.dtime.dt.time
    if 3<=xmonth<6:
        return '春'
    elif 6<=xmonth<9:
        return '夏'
    elif 9<=xmonth<12:
        return '秋'
    else:
        return '冬'
    
member['gwjjp'] = member['dtime'].map(get_gwjjp)
member.head()


# In[6]:


# 获取购买总积分、总金额、商品名称、消费时间等数据
data = member.groupby(["kh"], as_index=False).agg({'jf':'sum', 'je':'sum', 'spmc':'unique', 'dtime':'unique'})
print(data.shape, member.shape)
data.head()    


# In[7]:


# 合并得到会员的 积分，消费水平，购物类型的偏好，性别，入会时长，购物时间的偏好，购物季节偏好，年龄等数据
label =  pd.merge(data, member.loc[:, ['kh','xb','rhsc', 'age', 'gwsjp', 'gwjjp']],on='kh', how="left")
print(label.shape)
label.head()


# In[8]:


label.je[label.je<100] = 0
label.je[(100<=label.je)&(label.je<=200)] = 100
label.je[label.je>200] = 200
label.je[label.je==0] = '低消费'
label.je[label.je==100] = '中等消费'
label.je[label.je==200] = '高消费'


# In[9]:


# 去除卡号重复的数据
print(label.shape)
label.drop_duplicates(subset=['kh'], inplace=True)
print(label.shape)
label.head()


# In[10]:


# 获取消费次数
label['xfcs'] = label.apply(lambda x:x['dtime'].shape[0], axis=1)       # 消费次数
label.head()


# ### 任务3.4 建立用户画像

# In[11]:


label.columns = ['卡号', '积分', '消费水平', '商品名称', 'dtime', '性别', '入会时长', '年龄', '购物时间偏好', '购物季节偏好', '消费次数']
user_portrait = label[['卡号', '积分', '消费水平', '商品名称', '性别', '入会时长', '年龄', '购物时间偏好', '购物季节偏好', '消费次数']]
user_portrait.reset_index(drop=True, inplace=True)
user_portrait.head()


# In[12]:


user_portrait.性别 = '女' if [user_portrait.性别=='女'] else '男'
# user_portrait.性别[user_portrait.性别==0.0] = '女'
# user_portrait.性别[user_portrait.性别==1.0] = '男'
user_portrait.head()


# In[13]:


# 定义获取用户画像的函数
def get_user_message(kh):
    message = label[label.卡号==kh]
    message.head
    for i in message:
        if(i!='商品名称' and i!='dtime'):
            print(i,': ', message.iloc[0][i])


# In[14]:


# 测试
kh = '00075d60'
get_user_message(kh)
label[label.卡号==kh]


# In[ ]:





# In[ ]:





# In[ ]:




