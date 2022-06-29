
import pandas as pd

df = pd.df = pd.read_csv("IRIS.csv")

#Principal Component Analysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logr=LogisticRegression
pca=PCA(n_components=2)

df=pd.read_csv("IRIS.csv")
X = df.drop("species",axis=1)
Y= df["species"]

pca.fit(X)
X=pca.transform(X)

print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=0,test_size=0.3)