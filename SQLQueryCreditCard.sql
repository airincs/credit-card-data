/*
Credit Card Data Exploration
Source: https://www.kaggle.com/datasets/samuelcortinhas/credit-card-approval-clean-data
*/

--SELECT *
--FROM CreditCardProject..clean_dataset


-- Showing denied clients and their income, from greatest to least

--SELECT Income, Approved
--FROM CreditCardProject..clean_dataset
--WHERE Approved = 0
--ORDER BY Income + 0 DESC


-- Showing denied clients and their credit score, from greatest to least

--SELECT CreditScore, Approved
--FROM CreditCardProject..clean_dataset
--WHERE Approved = 0
--ORDER BY CreditScore + 0 DESC


-- Shows bank customers and whether or not they were approved

--SELECT Approved, COUNT(*) AS NumberofBankCustomers
--FROM CreditCardProject..clean_dataset
--WHERE BankCustomer = 1
--GROUP BY Approved


-- Shows approval percentage based on ethnicity

SELECT Ethnicity, SUM(ROUND(Approved, 2)) / COUNT(*) AS ApprovalPercentage
FROM CreditCardProject..clean_dataset
GROUP BY Ethnicity