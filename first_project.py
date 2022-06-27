import pandas as pd

df = pd.df = pd.read_csv("IRIS.csv")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logr = LogisticRegression()

X = df.drop(['species'], axis=1)
print(X)

Y = df["species"]
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.3)

train = logr.fit(X_train, Y_train)

Y_pred = logr.predict(X_test)

print(accuracy_score(Y_test, Y_pred))
