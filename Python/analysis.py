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
train = df

##printing out general information
print(train.describe(include="all"))
print(train.columns)
print(pd.isnull(train).sum())
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
#plt.show()
sns.barplot(x="Ethnicity", y="Approved", data=train)
#plt.show()
sns.barplot(x="Married", y="Approved", data=train)
#plt.show()
sns.barplot(x="PriorDefault", y="Approved", data=train)
#plt.show()
sns.barplot(x="Employed", y="Approved", data=train)
#plt.show()
sns.barplot(x="Industry", y="Approved", data=train)
#plt.show()
sns.barplot(x="BankCustomer", y="Approved", data=train)
#plt.show()
sns.barplot(x="Citizen", y="Approved", data=train)
#plt.show()
sns.barplot(x="CategoricalAge", y="Approved", data=train)
#plt.show()

#mapping categorical values to numerical values
industry_mapping = {'Industrials': 1, 'Materials': 2, 'CommunicationServices': 3, 'Transport': 4, 'InformationTechnology': 5,
                    'Financials': 6, 'Energy': 7, 'Real Estate': 8, 'Utilities': 9, 'Research': 10, 'ConsumerDiscretionary': 11,
                    'Education': 12, 'ConsumerStaples': 13, 'Healthcare': 14}
ethnicity_mapping = {'White': 1, 'Black': 2, 'Asian': 3, 'Latino': 4, 'Other': 5}
citizen_mapping = {'ByBirth': 1, 'ByOtherMeans': 2, 'Temporary': 3}
train['IndustryMap'] = train['Industry'].map(industry_mapping)
train['EthnicityMap'] = train['Ethnicity'].map(ethnicity_mapping)
train['CitizenMap'] = train['Citizen'].map(citizen_mapping)


##Model Prediction
print(train.isnull().any())
predictors = train.drop(['Approved', 'Industry', 'Ethnicity', 'Citizen', 'CategoricalAge'], axis=1)
target = train['Approved']
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size=0.20, random_state=0)

#logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr = LogisticRegression(solver='lbfgs', max_iter=10000)
lr.fit(x_train, y_train)
y_predict = lr.predict(x_val)
acc_lr = round(accuracy_score(y_predict, y_val) * 100, 2)
print("Logistic Regression: ", acc_lr)
#Logistic Regression: 86.23

#support vector machine
from sklearn.svm import SVC
svc = SVC(max_iter=10000)
svc.fit(x_train, y_train)
y_predict = svc.predict(x_val)
acc_svc = round(accuracy_score(y_predict, y_val) * 100, 2)
print("Support Vector Machine: ", acc_svc)
#Support Vector Machine: 66.67

#decision tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_predict = dt.predict(x_val)
acc_dt = round(accuracy_score(y_predict, y_val) * 100, 2)
print("Decision Tree: ", acc_dt)
#Decision Tree: 81.16

#k-nearest neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_predict = knn.predict(x_val)
acc_knn = round(accuracy_score(y_predict, y_val) * 100, 2)
print("KNN: ", acc_knn)
#KNN: 73.19

#gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
gb.fit(x_train, y_train)
y_predict = gb.predict(x_val)
acc_gb = round(accuracy_score(y_predict, y_val) * 100, 2)
print("Gradient Boosting: ", acc_gb)
#Gradient Boosting:

##Model Comparison
model_list = pd.DataFrame({'Model': ['Logistic Regression', 'SVM', 'Decision Tree', 'KNN', 'Gradient Boosting'],
                           'Score': [acc_lr, acc_svc, acc_dt, acc_knn, acc_gb]})
print(model_list.sort_values(by='Score', ascending=False))