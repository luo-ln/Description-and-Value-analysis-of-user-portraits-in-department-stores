#!/usr/bin/env python
# coding: utf-8

# ## 第四部分：会员用户细分模型和营销方案制定
# 流程如下
# ####  1：利用RFM模型对会员用户进行精细划分并分析不同群体带来的价值差异
# ####  2：针对不同的类型的群体制定相应的营销方案(如优惠券赠送、一元购、打折促销、捆绑销售等)
# 
# RFM模型具体指的是模型中特征：R(最近一次消费Recency)、F(消费频率Frequency)、M(消费金额Monetary)

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
member = data_task1[['kh', 'djsj', 'je', 'dtime']]

print(member.shape)
print('处理前缺失值数目是：\n', member.isnull().sum())
member.dropna(inplace=True)
print('\n处理后缺失值数目是：\n', member.isnull().sum())
print(member.shape)

member.loc[:,'dtime'] = pd.to_datetime(member.loc[:,'dtime'],infer_datetime_format=True) # 将年份转为datatime格式 
member.loc[:,'djsj'] = pd.to_datetime(member.loc[:,'djsj'],infer_datetime_format=True) # 将年份转为datatime格式 


# In[3]:


# 筛选出2017年的数据并进行部分预处理
df = member.set_index('djsj') # 将djsj设置为index，晒掉入会时长大于10或小于0的异常数据
df = df[:'2017']
df = df.reset_index()     # 重设索引
df = df.set_index('dtime') # 将dtime设置为index，选出最近一年的消费数据，即2017年的数据
df = df['2017']  
df = df.drop(df[(df.je<1)].index)   # 将消费金额为负数或金额过大的数据删除
df = df.drop(df[(df.je>3e3)].index)   
df = df.reset_index()    # 重设索引  
print(df.shape)
df.head()


# In[4]:


data = df.groupby(["kh"], as_index=False).agg({'je':'sum', 'dtime':'unique', 'djsj':'unique'})
print(data.shape, member.shape)
data.head()


# In[5]:


data['xfcs'] = data.apply(lambda x:x['dtime'].shape[0], axis=1)       # 消费次数
data['Recency_d'] = data.apply(lambda x:x['dtime'][-1], axis=1)       # 最近一次消费时间
data['djsj'] = data.apply(lambda x:x['djsj'][0], axis=1)              # 入会时长
data = data.drop(data[(data.xfcs>365)].index)                          # 去除消费次数超过365次的数据
data['dtime'] = data.apply(lambda x:x['dtime'].shape[0], axis=1)             # 消费频率(次/月)
data.head()    


# In[7]:


# 最近一次消费（Recency_d、月）、消费频率（dtime，次/月）、消费金额（je, 元/年）
data_now = pd.to_datetime('2018-01-01') # 自定义截止时间
data['Recency_d'] = (data_now - data.Recency_d).dt.days  # 最近一次购买是几天前
data['djsj'] = (data_now - data.djsj).dt.days/365        # 入会时间,按年计算
data['avg_je'] = (data.je)/12                            # 平均消费金额
data.head()


# In[8]:


for name in data.columns:
    print(name,"   ", data[name].min(),"   ", data[name].max())


# In[9]:


data_new = data.drop(data[(data.je> 99999)].index)  # 将总金额大于9999的异常值进行去除
data_new.reset_index(drop=True, inplace=True)   # 重设索引，并删除原来的索引值
data_new.head()


# In[10]:


data_new.rename(columns={'je':'消费总金额', 'dtime':'消费频率', 'djsj':'入会时长', 'xfcs':'消费次数', 'Recency_d':'最近一次消费', 'avg_je':'平均消费金额'}, inplace = True)
data_new.head()


# In[11]:


# 抽取出代表RFM的特征
train_data = data_new.iloc[:, [1,2,5]] # ['Monetary', 'Recency_d', 'Frequency']
#train_data = (train_data-train_data.min())/(train_data.max()-train_data.min()) # 归一化
train_data.head()


# In[12]:


# 评分
for name in train_data.columns:
    if name=="最近一次消费":
        ascending = False
    else:
        ascending = True
    train_data.sort_values(by=[name],inplace=True, ascending=ascending)
    train_data.reset_index(drop=True, inplace=True)
    n = train_data.shape[0]
    for i in range(1, 6):
        b = 0.2*n*(i-1)
        e = 0.2*n*i
        train_data.loc[b:e, name] = i


# In[15]:


for name in train_data.columns:
    print(name,"   ", train_data[name].min(),"   ", train_data[name].max())
train_data.head()


# In[16]:


train_data.info()


# ### 利用k-means进行聚类分析，并绘制雷达图

# In[17]:


from sklearn.cluster import KMeans 


# In[18]:


#需要进行的聚类类别数
k = 8
kmodel = KMeans(n_clusters = k) 
#训练模型
kmodel.fit(train_data) 

#查看聚类中心
print(kmodel.cluster_centers_) 
#查看各样本对应的类别
print(kmodel.labels_ )
print(kmodel.cluster_centers_.shape)


# In[19]:


# 简单打印结果
r1 = pd.Series(kmodel.labels_).value_counts() #统计各个类别的数目
r2 = pd.DataFrame(kmodel.cluster_centers_) #找出聚类中心

# 所有簇中心坐标值中最大值和最小值
max = r2.values.max()
min = r2.values.min()
r = pd.concat([r2, r1], axis = 1) #横向连接（0是纵向），得到聚类中心对应的类别下的数目
r.columns = list(train_data.columns) + [u'类别数目'] #重命名表头
 
# 绘图
fig=plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, polar=True)
center_num = r.values
feature = list(train_data.columns)
#feature = ['消费总金额', '消费频率', '入会时长', '消费次数', '最近一次消费', '平均消费金额']
N =len(feature)
for i, v in enumerate(center_num):
    # 设置雷达图的角度，用于平分切开一个圆面
    angles=np.linspace(0, 2*np.pi, N, endpoint=False)
    # 为了使雷达图一圈封闭起来，需要下面的步骤
    center = np.concatenate((v[:-1],[v[0]]))
    angles=np.concatenate((angles,[angles[0]]))
    # 绘制折线图
    ax.plot(angles, center, 'o-', linewidth=2, label = "第%d簇人群,%d人"% (i+1,v[-1]))
    # 填充颜色
    ax.fill(angles, center, alpha=0.25)
    # 添加每个特征的标签
    ax.set_thetagrids(angles * 180/np.pi, feature, fontsize=15)
    # 设置雷达图的范围
    ax.set_ylim(min-0.1, max+0.1)
    # 添加标题
    plt.title('客户群特征分析图', fontsize=20)
    # 添加网格线
    ax.grid(True)
    # 设置图例
    plt.legend(loc='upper right', bbox_to_anchor=(1.3,1.0),ncol=1,fancybox=True,shadow=True)
    
# 显示图形
#plt.savefig("%s类.png"%k)
plt.show()


# In[20]:


r['分数'] = r['消费总金额'] + r['消费频率'] + r['最近一次消费']
r.sort_values('分数', ascending=False, inplace=True)
tmp = r['类别数目']
tmp3 = tmp/r['类别数目'].sum()
ind = ['重要价值客户', '重要唤回客户', '重要深耕客户', '重要挽留客户', '深耕客户', '新客户', '一般维持客户', '流失客户']
tmp3.index = ind
print(tmp3)


# In[21]:


ax = tmp3.plot.barh(stacked=True,colormap = 'Blues_r') 
ax.figsize=(35, 35)
ax.set_xlabel('比例')  # 设置x轴标签
ax.set_ylabel('客户类型') # 设置y轴标签
fig = ax.get_figure()  # 用于保存图片
#fig.savefig('客户价值分类图.png',bbox_inches = 'tight')  # 保存为png格式


# In[ ]:





# In[ ]:




