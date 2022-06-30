import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2




lr =LinearRegression()

df =pd.read_csv("HousingData.csv")
# print(df.describe())

# to fill Nan spaces
for i in df.columns:
    df[i].fillna((df[i].mean()),inplace=True)
#print(df.head(10))

# x=df.iloc[:,['RM'],['PTRATIO'],['LTSTAT']]
x=df.drop(columns=['MEDV'],axis=1)
#print("All features",x)

y=df["MEDV"]
#print("medv",y)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.3)

train=lr.fit(x_train,y_train)

y_p =lr.predict(x_test)
# print(accuracy_score(y_test,y_p),normalise=False)
print(mean_squared_error(y_test,y_p))



# pca=PCA(n_components=2)
#
# pca.fit(x)
# x1=pca.transform(x)
#
# print(x1)
#
# X_train, X_test, Y_train, Y_test = train_test_split(x1,y,random_state=0,test_size=0.3)

# train1=lr.fit(X_train,Y_train)
#
# y_p =lr.predict(X_test)
# print(accuracy_score(Y_test,y_p))

'''
#Identifying Outliers
from matplotlib import pyplot as plt
import seaborn as sns
sns.boxplot(df['ZN'])
plt.show()
'''
# x2=df.isnull().sum()
# print(x2)

# import seaborn as sns
# correlation_matrix = df.corr().round(2)
# # annot = True to print the values inside the square
# x3=sns.heatmap(data=correlation_matrix, annot=True)
# print(x3)

# print(df.hist(figsize=(12,12)))

