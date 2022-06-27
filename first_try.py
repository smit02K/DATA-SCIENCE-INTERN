import pandas as pd

df = pd.read_csv("IRIS.csv")

# print(df)

# print(df.head(10)) #gives starting 10 rows of all columns
# print(df.tail(20))


print(df.columns.values) #gives list of features

# print(df.describe())

# print(df.iloc[0:100])   #gives desired rows of all columns
# print(df.iloc[3:5])


# print(df.loc[:,["petal_length","species"]]) #gives desired rows of desired columns


# print(df[df['petal_width']==0.2]) #gives all the rows of the columns whose value is greater than 0.1
# print(df[df['species']=='Iris-setosa'])

print(df[45:75])  #gives 45 to 74 ka rows of all columns
