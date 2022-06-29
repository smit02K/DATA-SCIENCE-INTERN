import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

df=pd.read_csv("IRIS.csv")
X = df.drop("species",axis=1)
Y= df["species"]




#Feature Selection 1
bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featuresScores = pd.concat([dfcolumns, dfscores], axis=1)
featuresScores.columns = ['Specs', 'Score']

print(featuresScores)

#Feature Selection 2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)

feat_importance = pd.Series(model.feature_importances_, index=X.columns)
feat_importance.nlargest(4).plot(kind = 'barh')
plt.show()
