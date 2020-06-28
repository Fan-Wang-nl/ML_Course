#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import pandas model as a tool to read data
import pandas as pd

#利用pandas的read_csv模块直接从互联网手机泰坦尼克号乘客数据
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

titanic.head()


# In[5]:


titanic.info()


# In[7]:


#特征选择，sex，age，pclass这些特征很有可能是决定幸免与否的关键因素
X=titanic[['pclass','age','sex']]
y=titanic['survived']


# In[8]:


X.info()


# In[9]:


#填充age缺失值，使用平均数或中位数
X['age'].fillna(X['age'].mean(),inplace=True)
#查看数据特征
X.info()


# In[10]:


#数据分割,拆分训练集和测试集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)

#特征转换
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
print(vec.feature_names_)


# In[11]:


#对预测数据进行同样的特征转换
X_test = vec.transform(X_test.to_dict(orient='record'))


# In[12]:


#导入决策树模型并对测试特征数据进行预测
from sklearn.tree import DecisionTreeClassifier
#使用默认配置初始化决策树分类器
dtc = DecisionTreeClassifier()
#训练数据进行模型学习
dtc.fit(X_train,y_train)
#决策树模型对特征数据进行预测
y_predict=dtc.predict(X_test)


# In[13]:


#模型评估
from sklearn.metrics import classification_report
#输出预测准确性
print(dtc.score(X_test,y_test))
#输出更加详细的分类性能
print(classification_report(y_predict,y_test,target_names=['died','survived']))


# In[14]:


#使用随机森林分类器进行集成模型的训练以及预测分析
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_pred=rfc.predict(X_test)
#输出随机森林分类器在测试集上的分类准确性性
print(rfc.score(X_test,y_test))
print(classification_report(rfc_y_pred,y_test))



#使用梯度提升决策树进行集成O型的训练及预测分析
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_pred=gbc.predict(X_test)
#输出梯度提升树在测试集上的分类准确性
print(gbc.score(X_test,y_test))
print(classification_report(gbc_y_pred,y_test))

