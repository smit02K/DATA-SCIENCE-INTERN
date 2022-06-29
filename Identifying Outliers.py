#Identifying Outliers by ploting

import pandas as pd

df = pd.df = pd.read_csv("IRIS.csv")

# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.boxplot(df['sepal_length'])
# plt.show()


#Identifying Outliers using Interquantile Range
print(df['sepal_length'])
Q1 = df['sepal_length'].quantile(0.25)
Q3 = df['sepal_length'].quantile(0.75)

IQR = Q3 - Q1
print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR

print(upper)
print(lower)

out1=df[df['sepal_length'] < lower].values
out2=df[df['sepal_length'] > upper].values

df['sepal_length'].replace(out1,lower,inplace=True)
df['sepal_length'].replace(out2,upper,inplace=True)

print(df['sepal_length'])


