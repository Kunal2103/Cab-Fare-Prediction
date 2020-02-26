#!/usr/bin/env python
# coding: utf-8

# # Project 2 ~ Cab Fare Prediction

# In[1]:


#loading libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split







# In[2]:


#set working directory
os.chdir("C:/python")


# In[3]:


#check wd
os.getcwd()


# In[4]:


#load the data
cab = pd.read_csv("train_cab.csv")


# #####data exploration #####

# In[5]:


#variables and observations
cab.shape


# In[6]:


#find datatypes
cab.dtypes


# In[7]:


#converting 'fare_amount' to numeric
cab['fare_amount']=pd.to_numeric(cab['fare_amount'], errors="coerce")


# In[8]:


#extracting different features from "pickup_datetime"
cab['pickup_datetime']=pd.to_datetime(cab['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC', errors="coerce")


# In[9]:


cab['year'] = cab['pickup_datetime'].dt.year
cab['month'] = cab['pickup_datetime'].dt.month
cab['weekdays'] = cab['pickup_datetime'].dt.weekday
cab['days'] = cab['pickup_datetime'].dt.day
cab['time'] = cab['pickup_datetime'].dt.hour


# In[10]:


#removing unwanted features
cab = cab.drop(['pickup_datetime'], axis=1)


# ############   DATA PRE-PROCSSING   ##############

# In[11]:


###### Missing Value Analysis ######

#replacing negative values of 'fare_amount' with zero
cab['fare_amount'][ cab['fare_amount'] < 0 ] = 0


# In[23]:


#find the count of missing values
cab.isna().sum()


# In[22]:


#replacing zeroes to nan
cab['fare_amount'] = cab['fare_amount'].replace(0, np.nan)
cab['pickup_longitude'] = cab['pickup_longitude'].replace(0, np.nan)
cab['pickup_latitude'] = cab['pickup_latitude'].replace(0, np.nan)
cab['dropoff_longitude'] = cab['dropoff_longitude'].replace(0, np.nan)
cab['dropoff_latitude'] = cab['dropoff_latitude'].replace(0, np.nan)
cab['passenger_count'] = cab['passenger_count'].replace(0, np.nan)
cab['month'] = cab['month'].replace(0, np.nan)
cab['days'] = cab['days'].replace(0, np.nan)
cab['year'] = cab['year'].replace(0, np.nan)


# In[24]:


#find the count of missing values
cab.isna().sum()


# In[25]:


#create a dataframe with no of missing values
missing_val = pd.DataFrame(cab.isnull().sum())


# In[26]:


#reset index
missing_val = missing_val.reset_index()


# In[27]:


#rename columns
missing_val = missing_val.rename(columns = {'index': 'Variables', 0: "Missing-percentage"})


# In[28]:


#calculate percentage
missing_val["Missing-percentage"] = (missing_val['Missing-percentage']/len(cab))*100


# In[29]:


#sort in descending order
missing_val = missing_val.sort_values('Missing-percentage', ascending = False).reset_index(drop = True)


# In[30]:


missing_val


# In[35]:


#observe a value
cab['fare_amount'].loc[1]


# In[32]:


#convert the value to nan
cab['fare_amount'].loc[1] = np.nan


# In[23]:


#Actual value = 16.9
#Mean = 15.01
#Median = 8.5


# In[34]:


#impute with mean 
cab['fare_amount'] = cab['fare_amount'].fillna(cab['fare_amount'].mean())
cab['pickup_longitude'] = cab['pickup_longitude'].fillna(cab['pickup_longitude'].mean())
cab['pickup_latitude'] = cab['pickup_latitude'].fillna(cab['pickup_latitude'].mean())
cab['dropoff_longitude'] = cab['dropoff_longitude'].fillna(cab['dropoff_longitude'].mean())
cab['dropoff_latitude'] = cab['dropoff_latitude'].fillna(cab['dropoff_latitude'].mean())
cab['passenger_count'] = cab['passenger_count'].fillna(cab['passenger_count'].mean())
cab['year'] = cab['year'].fillna(cab['year'].mean())
cab['days'] = cab['days'].fillna(cab['days'].mean())
cab['month'] = cab['month'].fillna(cab['month'].mean())
cab['weekdays'] = cab['weekdays'].fillna(cab['weekdays'].mean())
cab['time'] = cab['time'].fillna(cab['time'].mean())


# In[29]:


#impute with median
cab['fare_amount'] = cab['fare_amount'].fillna(cab['fare_amount'].median())


# In[36]:


cab.isna().sum()


# In[37]:


cab['month'] = cab['month'].astype(int)
cab['time']= cab['time'].astype(int)
cab['weekdays']= cab['weekdays'].astype(int)


# In[38]:


#converting to objects
cab['month'] = cab['month'].astype(object)
cab['time'] = cab['time'].astype(object)
cab['weekdays'] = cab['weekdays'].astype(object)


# ####outlier analysis####
# 
# 

# In[39]:


#plot boxplot
get_ipython().run_line_magic('matplotlib', 'inline')

plt.boxplot(cab['fare_amount'])


# In[40]:


#select only numeric
cnames = cab.select_dtypes(include=np.number)


# In[41]:


#finding & removing outliers
for i in cnames:
    #print(i)
    q75, q25 = np.percentile(cab.loc[:,i], [75,25])
    iqr = q75 - q25
    
    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    #print(min)
    #print(max)
 
    cab = cab.drop(cab[cab.loc[:,i] < min].index)
    cab = cab.drop(cab[cab.loc[:,i] > max].index)


# In[42]:


cab.shape


# ######Feature Selection ######
#  

# In[43]:


#select only numeric
cnames = cab.select_dtypes(include=np.number)


# In[44]:


#heatmap
#dimensions of heatmap
f, ax = plt.subplots(figsize=(100, 7))

#correlation matrix
corr = cnames.corr()

#plot
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
           square=True, ax=ax)


# In[ ]:


#no need to remove any numerical feature (no dependency)


# ######### Feature scaling #########
# 

# In[45]:


#plot histogram to check normalisation
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(cab['fare_amount'], bins='auto')                             ##not normally distributed


# In[46]:


#select only numeric
cnames = cab.select_dtypes(include=np.number)


# In[47]:


cnames = cnames.drop(['fare_amount'], axis=1)


# In[48]:


#normalisation
for i in cnames:
    print(i)
    cab[i] = (cab[i] - (cab[i].min()))/((cab[i].max()) - (cab[i].min()))


# ############################# Modelling ###################################
# 

# In[50]:


#MAPE function
def MAPE(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true))*100
    return mape


# In[51]:


# divide into train & test
sample = np.random.rand(len(cab)) < 0.9

train = cab[sample]
test = cab[~sample]


# ################### Linear regression ###################
# 

# In[52]:


#run model
model = sm.OLS(train.iloc[:,0], train.iloc[:,1:11].astype(float)).fit()


# In[53]:


#prediction on test data
predictions_LR = model.predict(test.iloc[:,1:11])


# In[54]:


#evaluation
Error_LR = round(MAPE(test.iloc[:,0], predictions_LR), 3)
Accuracy_LR = round(100 - Error_LR, 3)


# In[55]:


print("Error_LR = ", Error_LR)
print("Accuracy_LR = ", Accuracy_LR)


# In[56]:


RMSE = sqrt(mean_squared_error(test.iloc[:,0],predictions_LR ))
print(RMSE)


# ################## Decision tree ###################
# 

# In[57]:


#run model
fit_DT = DecisionTreeRegressor(max_depth=2).fit(train.iloc[:,1:11],train.iloc[:,0])


# In[58]:


#prediction
predictions_DT = fit_DT.predict(test.iloc[:,1:11])


# In[59]:


#evaluation
Error_DT = round(MAPE(test.iloc[:,0], predictions_DT), 3)
Accuracy_DT = round(100 - Error_DT, 3)


# In[60]:


print("Error_DT = ", Error_DT)
print("Accuracy_DT = ", Accuracy_DT)


# In[60]:


RMSE = sqrt(mean_squared_error(test.iloc[:,0], predictions_DT))
print(RMSE)


# ############### Random forest #################
# 

# In[136]:


#independent & dependent variables
x = cab.iloc[:,0:11]
y = cab.iloc[:,0]


# In[137]:


#splitting data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[138]:


#run model
RF_model = RandomForestRegressor(n_estimators = 100, random_state = 0)


# In[139]:


#summary
RF_model.fit(x_train,y_train)


# In[140]:


#prediction
RF_predictions = RF_model.predict(x_test)


# In[141]:


#evaluation
Error_RF = round(MAPE(x_test.iloc[:,0], RF_predictions), 3)
Accuracy_RF = round(100 - Error_RF, 3)


# In[142]:


print("Error_RF = ", Error_RF)
print("Accuracy_RF = ", Accuracy_RF)


# In[68]:


RMSE = sqrt(mean_squared_error(x_test.iloc[:,0], RF_predictions))
print(RMSE)


# ##########prediction on large test data ############

# In[70]:


#importing large test data
cab_test = pd.read_csv("test.csv")


# In[72]:


#find datatypes
cab_test.dtypes


# In[73]:


#extracting different features from "pickup_datetime"
cab_test['pickup_datetime']=pd.to_datetime(cab_test['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC', errors="coerce")


# In[75]:


cab_test['year'] = cab_test['pickup_datetime'].dt.year
cab_test['month'] = cab_test['pickup_datetime'].dt.month
cab_test['weekdays'] = cab_test['pickup_datetime'].dt.weekday
cab_test['days'] = cab_test['pickup_datetime'].dt.day
cab_test['time'] = cab_test['pickup_datetime'].dt.hour


# In[77]:


#removing unwanted features
cab_test = cab_test.drop(['pickup_datetime'], axis=1)


# #########data pre-processing############

# In[78]:


###### Missing Value Analysis ######

#find the count of missing values
cab_test.isna().sum()


# In[79]:


#replacing zeroes to nan
cab_test['pickup_longitude'] = cab_test['pickup_longitude'].replace(0, np.nan)
cab_test['pickup_latitude'] = cab_test['pickup_latitude'].replace(0, np.nan)
cab_test['dropoff_longitude'] = cab_test['dropoff_longitude'].replace(0, np.nan)
cab_test['dropoff_latitude'] = cab_test['dropoff_latitude'].replace(0, np.nan)
cab_test['passenger_count'] = cab_test['passenger_count'].replace(0, np.nan)
cab_test['month'] = cab_test['month'].replace(0, np.nan)
cab_test['days'] = cab_test['days'].replace(0, np.nan)
cab_test['year'] = cab_test['year'].replace(0, np.nan)


# In[80]:


#find the count of missing values
cab_test.isna().sum()


# In[81]:


#no missing values found in the test data


# In[82]:


#converting back some features to factors
cab_test['month'] = cab_test['month'].astype(object)
cab_test['weekdays'] = cab_test['weekdays'].astype(object)
cab_test['time'] = cab_test['time'].astype(object)


# ##########Feature scaling #########
# 

# In[84]:


#plot histogram to check normalisation
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(cab_test['passenger_count'], bins='auto')


# In[85]:


#select only numeric
cnames = cab_test.select_dtypes(include=np.number)


# In[87]:


#normalisation
for i in cnames:
    print(i)
    cab_test[i] = (cab_test[i] - (cab_test[i].min()))/((cab_test[i].max()) - (cab_test[i].min()))


# In[129]:


#prediction
predictions_test = model.predict(cab_test)


# In[130]:


#importing original test dataset
cab_results = pd.read_csv("test.csv")


# In[131]:


#columnbind target results with test data
cab_results['predictions_test'] = pd.DataFrame(predictions_test)


# In[132]:


#saving output in excel format
cab_results.to_csv("Fare amount results - Python.csv", index = False)


# In[ ]:




