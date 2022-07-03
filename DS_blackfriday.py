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
selector = ExtraTreesRegressor()
selector.fit(X, Y)
feature_imp = selector.feature_importances_
X.drop(['Gender', 'City_Category', 'Marital_Status'], axis = 1, inplace = True)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
lin_reg = LinearRegression()
knn = KNeighborsRegressor()
dec_tree = DecisionTreeRegressor()
ran_for = RandomForestRegressor()
dtc=DecisionTreeClassifier()

print("MEAN SQUARED ERRORS")
lin_reg.fit(X_train, Y_train)
Y_pred_lin_reg = lin_reg.predict(X_test)
print("Linear Regression: ",mean_squared_error(Y_test, Y_pred_lin_reg))


knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
print("KNN regression: ",mean_squared_error(Y_test, Y_pred_knn))


dec_tree.fit(X_train, Y_train)
Y_pred_dec = dec_tree.predict(X_test)
print("Decision tree regression: ",mean_squared_error(Y_test, Y_pred_dec))


ran_for.fit(X_train, Y_train)
Y_pred_ran_for = ran_for.predict(X_test)
print("Random forest regression: ",mean_squared_error(Y_test, Y_pred_ran_for))


'''
MEAN SQUARED ERRORS
Linear Regression:  22044840.101023477
KNN regression:  10469708.932551857
Decision tree regression:  9843766.502839973
Random forest regression:  9058790.950768305
'''
