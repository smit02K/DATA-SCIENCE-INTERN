
import pandas as pd

#Oversampling & Under Sampling
from imblearn.over_sampling import RandomOverSampler   #Random OverSampling

df = pd.df = pd.read_csv("IRIS.csv")

X = df.drop(['species'], axis=1)
print(X)

Y = df["species"]
print(Y)



ros = RandomOverSampler(random_state=0)
X, Y = ros.fit_resample(X,Y)

from imblearn.over_sampling import SMOTE     #Synthetic Minority Oversampling (Smote)
sms = SMOTE(random_state=0)
X, Y = sms.fit_resample(X,Y)

from imblearn.under_sampling import RandomUnderSampler    #Random UnderSampling
rus=RandomUnderSampler (random_state=0)
X, Y=rus.fit_resample(X,Y)  