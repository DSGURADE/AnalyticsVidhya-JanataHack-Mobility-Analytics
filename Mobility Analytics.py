#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

# libraby for linear algebra
import numpy as np 

# library for data processing
import pandas as pd 

# library for data visualization
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib import style


# ### Data Reading

# In[2]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.head()


# In[3]:


# Top 5 records of train dataframe
train.head()


# In[4]:


# Check the column-wise info of the train dataframe
train.info()


# In[5]:


# Get a summary of the train dataframe using 'describe()'
train.describe()


# In[6]:


# Check the number of rows and columns in the dataframes
print(train.shape)
print(test.shape)


# In[7]:


sns.countplot(train['Surge_Pricing_Type'])
plt.show()


# In[8]:


# See the Surge_Pricing_Type with the Gender
pd.crosstab(train['Gender'],train['Surge_Pricing_Type'] ).plot(kind="bar",figsize=(15,6))
plt.title('Frequency of Surge_Pricing_tyoe variable with Gender')
plt.xticks(rotation=0)
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.show()


# In[9]:


# See the Surge_Pricing_Type with the Gender
pd.crosstab(train['Customer_Since_Months'],train['Surge_Pricing_Type'] ).plot(kind="bar",figsize=(15,6))
plt.title('Frequency of Customer_Since_Months variable with Gender')
plt.xticks(rotation=0)
plt.xlabel('Customer_Since_Months')
plt.ylabel('Frequency')
plt.show()


# In[10]:


# Outlier analysis for Type_of_cab with Trip distance variable
sns.boxplot(x='Type_of_Cab',y = 'Trip_Distance',data=train)
plt.show()


# In[11]:


# Outlier analysis for Customer_rating with trip distance variable
sns.boxplot(x='Type_of_Cab',y = 'Customer_Rating',data=train)
plt.show()


# In[12]:


sns.countplot(train['Type_of_Cab'])
plt.show()


# In[13]:


sns.countplot(train['Confidence_Life_Style_Index'])
plt.show()


# ### Data cleaning and Data Prepration

# In[14]:


# Get the column-wise Null count
train.isnull().sum()


# In[15]:


# Get the column-wise Null Percentage
round(100*(train.isnull().sum()/len(train)),2)


# In[16]:


# Filling null values train dataset with mean of that column
train['Type_of_Cab'].fillna("B", inplace=True)
train['Customer_Since_Months'].fillna(round(train['Customer_Since_Months'].mean()),inplace=True)
train['Life_Style_Index'].fillna(train['Life_Style_Index'].mean(),inplace=True)
train['Confidence_Life_Style_Index'].fillna("B",inplace=True)
train['Var1'].fillna(round(train['Var1'].mean()),inplace=True)


# In[17]:


# Get the column-wise Null count
test.isnull().sum()


# In[18]:


# Get the column-wise Null Percentage
round(100*(test.isnull().sum()/len(test)),2)


# In[19]:


# Filling null values test dataset with mean of that column
test['Type_of_Cab'].fillna("B", inplace=True)
test['Customer_Since_Months'].fillna(round(test['Customer_Since_Months'].mean()),inplace=True)
test['Life_Style_Index'].fillna(test['Life_Style_Index'].mean(),inplace=True)
test['Confidence_Life_Style_Index'].fillna("B",inplace=True)
test['Var1'].fillna(round(test['Var1'].mean()),inplace=True)


# ### Label Encoding

# In[20]:


categorical_var = ['Type_of_Cab','Confidence_Life_Style_Index','Destination_Type','Gender']

# Label Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

for x in categorical_var:
    if train[x].dtype == type(object):
        encoder = LabelEncoder()
        encoder.fit(list(set(list(train[x]) + list(test[x]))))
        train[x] = encoder.transform(train[x])
        test[x] = encoder.transform(test[x])


# In[21]:


train.head()


# In[22]:


# plot training dataset heatmap
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# In[23]:


# Dropping unnecessary columns from train dataset
train.drop(['Trip_ID','Gender'], axis=1, inplace=True)
test.drop(['Trip_ID','Gender'], axis=1, inplace=True)


# In[24]:


X = train.drop(['Surge_Pricing_Type'],axis=1)
y = train['Surge_Pricing_Type']


# In[25]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
test_data = sc.transform(test.copy().values)


# In[26]:


# XGB Classifier
from xgboost import XGBClassifier

xgb = XGBClassifier( learning_rate =0.1,
 n_estimators=112,
 max_depth=9,
 min_child_weight=5,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.6,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=13,
 reg_lambda=5,
# max_delta_step=1,
 alpha=0,
 base_score=0.5,
 seed=1029)

xgb.fit(X_train, y_train)

# Predicting the Test set results
y_pred = xgb.predict(X_test)  

# Accuracy of XGB model
accuracy_xgb = round(xgb.score(X_train, y_train) * 100, 2)
print("Accuracy score of XGB algorithm is:", accuracy_xgb)


# In[27]:


# Predicting the Test set results
test_pred = xgb.predict(test_data)


# In[28]:


# load Trip_id of test dataset
test_Trip_ID = pd.read_csv('test.csv')['Trip_ID']
print(test_Trip_ID.shape)


# In[29]:


# save results to csv
submission_file = pd.DataFrame({'Trip_ID': test_Trip_ID, 'Surge_Pricing_Type': test_pred})
submission_file = submission_file[['Trip_ID','Surge_Pricing_Type']] 
submission_file.to_csv('Final_Solution.csv', index=False)


# In[ ]:




