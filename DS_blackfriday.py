import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#importing the dataset
from sklearn.metrics import accuracy_score, mean_squared_error

df_train = pd.read_csv('train.csv')
# print(df_train.head())

#  import the test data
df_test = pd.read_csv('test.csv')
# print(df_test.head())

#MErge both train and test data
df=df_train.append(df_test)
# print(df.head(10))

##Basic
# print(df.info())
#print(df.describe())

df.drop(['User_ID'],axis=1,inplace=True)
#print(df.head())

'''
#not effective-. use map
df['Gender']=pd.get_dummies(df['Gender'],drop_first=1)
'''

##HAndling categorical feature Gender
df['Gender']=df['Gender'].map({'F':0,'M':1})
#print(df.head())

## Handle categorical feature Age
#print(df['Age'].unique())


# map age
df['Age']=df['Age'].map({'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7})
#print(df.head())

'''
##second technqiue
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
df['Age'] = label_encoder.fit_transform(df['Age'])

df['Age'].unique()
'''

##fixing categorical City_categort
df_city=pd.get_dummies(df['City_Category'],drop_first=True)
#print(df_city.head())

##fixing categorical City_categort
df_city=pd.get_dummies(df['City_Category'],drop_first=True)
#print(df_city.head())

df=pd.concat([df,df_city],axis=1)
#print(df.head())

df.drop('City_Category',axis=1,inplace=True)
#print(df.head())


## Missing Values
# print(df.isnull().sum())

# ## Focus on replacing missing values
# print(df['Product_Category_2'].unique())
#
# print(df['Product_Category_2'].value_counts())
#
# print(df['Product_Category_2'].mode()[0])

## Replace the missing values with mode
df['Product_Category_2']=df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0])
# print(df['Product_Category_2'].isnull().sum())


## Focus on replacing missing values
# print(df['Product_Category_3'].unique())
#
# print(df['Product_Category_3'].value_counts())
#
# print(df['Product_Category_3'].mode()[0])

## Replace the missing values with mode
df['Product_Category_3']=df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0])
# print(df['Product_Category_3'].isnull().sum())
# print(df.head())

#print(df['Stay_In_Current_City_Years'].unique())
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].str.replace('+','')
#print(df.head())

#print(df.info())
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].astype(int)
#print(df.info())

df['B']=df['B'].astype(int)
df['C']=df['C'].astype(int)
#print(df.info())

##Feature Scaling
df_test=df[df['Purchase'].isnull()]
df_train=df[~df['Purchase'].isnull()]
X=df_train.drop('Purchase',axis=1)

#print(X.head())

y=df_train['Purchase']
print(X.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train.drop('Product_ID',axis=1,inplace=True)
X_test.drop('Product_ID',axis=1,inplace=True)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

