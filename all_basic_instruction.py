import pandas as pd
df=pd.read_csv("IRIS.csv")
# print dataset values
# print(df)
# first 10 values from dataset
print(df.head(10))
# last 10 values from dataset
print(df.tail(10))
# which column (features) are there
print(df.columns.values)
# used for describing dataset
print(df.describe())
# used for printing specific columns
print(df.loc[:, ["petal_length", "species"]])
# used for printing according to condition
print(df[df['petal_width'] > 0.1])
# print(df[df.loc[:, ['petal_width']] > 0.1])
# used for printing rows from 0 to 99
print(df.loc[0:99, ["petal_length", "species"]])
# used for printing rows from 0 to all ahead
print(df.loc[0:, ["petal_length", "species"]])
# used for printing specific row (row 0 here)
print(df.loc[0, ["petal_length", "species"]])
print(df[45:75])
print(df[df['species'] == 'Iris-satosa'])
