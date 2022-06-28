import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


x_df = pd.read_csv("IRIS.csv")
# print(x_train)
y_df = pd.read_csv("IRIS.csv")

X=x_df.drop(columns= ["species"],axis=1)
print(X)

Y=y_df['species']
print(Y)

bestfeatures = SelectKBest(score_func=chi2 ,k='all')
fit =bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis =1)
featureScores.columns =['Specs','Score']
print(featureScores)


#           Specs       Score
# 0  sepal_length   10.817821
# 1   sepal_width    3.594499
# 2  petal_length  116.169847
# 3   petal_width   67.244828