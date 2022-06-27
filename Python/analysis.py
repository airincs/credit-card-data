import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

##reading in data
df = pd.read_csv("./credit_card.csv")
train, test = train_test_split(df, test_size=0.20)

##printing out general information
#print(train.describe(include="all"))
#print(train.columns)
#print(pd.isnull(train).sum())
#Data has already been cleaned; therefore, there are no missing values

##Feature Engineering
print(train[['Gender', "Approved"]].groupby(['Gender'], as_index=False).mean())
print(train[['Ethnicity', "Approved"]].groupby(['Ethnicity'], as_index=False).mean())
print(train[['Married', "Approved"]].groupby(['Married'], as_index=False).mean())
print(train[['PriorDefault', "Approved"]].groupby(['PriorDefault'], as_index=False).mean())
print(train[['Employed', "Approved"]].groupby(['Employed'], as_index=False).mean())
print(train[['Industry', "Approved"]].groupby(['Industry'], as_index=False).mean())
print(train[['BankCustomer', "Approved"]].groupby(['BankCustomer'], as_index=False).mean())
print(train[['Citizen', "Approved"]].groupby(['Citizen'], as_index=False).mean())
print(train[['DriversLicense', "Approved"]].groupby(['DriversLicense'], as_index=False).mean())

train['CategoricalAge'] = pd.cut(train['Age'], 5)
print(train[['CategoricalAge', "Approved"]].groupby(['CategoricalAge'], as_index=False).mean())

#interesting variables: gender, ethnicity, married, priordefault, employed,industry, bankcustomer,citizen, categoricalage
sns.barplot(x="Gender", y="Approved", data=train)
plt.show()
sns.barplot(x="Ethnicity", y="Approved", data=train)
plt.show()
sns.barplot(x="Married", y="Approved", data=train)
plt.show()
sns.barplot(x="PriorDefault", y="Approved", data=train)
plt.show()
sns.barplot(x="Employed", y="Approved", data=train)
plt.show()
sns.barplot(x="Industry", y="Approved", data=train)
plt.show()
sns.barplot(x="BankCustomer", y="Approved", data=train)
plt.show()
sns.barplot(x="Citizen", y="Approved", data=train)
plt.show()
sns.barplot(x="CategoricalAge", y="Approved", data=train)
plt.show()