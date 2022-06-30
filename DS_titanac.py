import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv("Titanic dataset.csv")
print(df)

#bos = load_boston
lreg = LinearRegression ()

# print(df.head(4))
# print(df.describe())

x = df.drop('PassengerId',axis=1)
x = x.drop('Survived', axis=1)
x = x.drop('Name',axis=1)
x = x.drop('Ticket',axis=1)
x = x.drop('Cabin',axis=1)
x = x.drop('Embarked',axis=1)
x = x.drop('Parch',axis=1)
x = x.drop('Sex',axis=1)

y = df['Survived']

x['Age'].fillna((x['Age'].mean()), inplace=True)
x['Fare'].fillna((x['Fare'].mean()), inplace=True)
# print(x.info())
# print(y.info())

print(x)


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.2)

reg_train = lreg.fit(x_train,y_train)
reg_pred = lreg.predict(x_test)

print('Mean Squared Error=',mean_squared_error(y_test,reg_pred))
