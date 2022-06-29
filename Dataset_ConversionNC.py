import pandas as pd
#Dataset Conversion
# Numerical to Categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('IRIS.csv')
rf= RandomForestClassifier()

df['sepal_length']=pd.cut(df['sepal_length'], 3, labels=['0', '1', '2'])
df['sepal_width']=pd.cut(df['sepal_width'], 3, labels=['0', '1', '2'])
df['petal_length']=pd.cut(df['petal_length'], 3, labels=['0', '1', '2'])
df['petal_width']=pd.cut(df['petal_width'], 3, labels=['0', '1', '2'])


df=pd.read_csv("IRIS.csv")
X = df.drop("species",axis=1)
Y= df["species"]

print(Y)
le=LabelEncoder()
le.fit(Y)
Y = le.transform(Y)
print(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0,test_size=0.2)

rf.fit(X_train,Y_train)
y_pred=rf.predict(X_test)
print('Random Forest: ', accuracy_score(Y_test,y_pred))


#Categorical to Numerical
le = LabelEncoder()
le.fit(Y)
Y = le.transform(Y)