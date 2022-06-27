# 💳Credit Card Analysis with SQL & Python💳

![MicrosoftSQLServer](https://img.shields.io/badge/Microsoft%20SQL%20Sever-CC2927?style=for-the-badge&logo=microsoft%20sql%20server&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)


### Project Details

![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)
- Source: https://www.kaggle.com/datasets/samuelcortinhas/credit-card-approval-clean-data
- Microsoft SQL Server was used initially to gather some insights into the data. After SQL, Python was used to analyze the data and to create several models.

## Purpose? Why?
This goal of this analysis was to see what variables most impact someone's chances at getting approved for a credit card. After some feature exploring, deciding on a most effective predictive model was the next step.

## Results
![alt text](https://github.com/airincs/credit-card-data/blob/main/images/Results.png?raw=true)
We can conclude that a Logistic Regression model would be the most fitting.

Some interesting variables:
![alt text](https://github.com/airincs/credit-card-data/blob/main/images/CategoricalAge.png?raw=true)
![alt text](https://github.com/airincs/credit-card-data/blob/main/images/Employed.png?raw=true)
![alt text](https://github.com/airincs/credit-card-data/blob/main/images/Ethnicity.png?raw=true)
![alt text](https://github.com/airincs/credit-card-data/blob/main/images/Industry.png?raw=true)
![alt text](https://github.com/airincs/credit-card-data/blob/main/images/PriorDefault.png?raw=true)

From the Correlation Matrix:
Looking at the Approval column, we can see what variables have the most influence on getting approved or not.
- Variables that positively influence approval chances: PriorDefault, Employment, CreditScore, and YearsEmployed
- Variables that negatively influence approval chances: Industry and Ethnicity
