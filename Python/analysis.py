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

##prints out general information
#print(train.describe(include="all"))
#print(train.columns)
#print(pd.isnull(train).sum())
#Data has already been cleaned; therefore, there are no missing values

