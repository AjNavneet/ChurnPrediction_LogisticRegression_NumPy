# Logistic Regression using Numpy - Churn Classification

## Business Objective

In this project, we aim to predict customer churn for a service-providing company called XYZ. Churned customers are those who have decided to end their relationship with the company. The company wants to know if the customers will renew their subscription for the coming year or not.

---

## Data Description

The dataset consists of approximately 2000 rows and 16 columns with the following features:

1. Year
2. Customer_id (unique id)
3. Phone_no (customer phone no)
4. Gender (Male/Female)
5. Age
6. No of days subscribed (the number of days since the subscription)
7. Multi-screen (does the customer have a single/multiple screen subscription)
8. Mail subscription (customer receives mails or not)
9. Weekly mins watched (number of minutes watched weekly)
10. Minimum daily mins (minimum minutes watched)
11. Maximum daily mins (maximum minutes watched)
12. Weekly nights max mins (number of minutes watched at night time)
13. Videos watched (total number of videos watched)
14. Maximum_days_inactive (days since inactive)
15. Customer support calls (number of customer support calls)
16. Churn (1-Yes, 0-No)

---

## Aim

The goal of this project is to build a logistic regression learning model using NumPy on the given dataset to determine whether the customer will churn or not.

---

## Tech Stack

- Language: Python
- Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn, pickle, imblearn, statsmodels

---

## Approach

1. Import the required libraries and read the dataset.
2. Perform Exploratory Data Analysis (EDA) with data visualization.
3. Conduct Feature Engineering by dropping unwanted columns.
4. Build a Logistic Regression Model using the statsmodels library.
5. Split the dataset into a training and testing set.
6. Validate the model's predictions, including accuracy score, confusion matrix, ROC and AUC, recall score, precision score, and F1-score.
7. Handle the unbalanced data using various methods, including balanced weights, random weights, and adjusting imbalanced data using SMOTE.
8. Perform feature selection with different methods, such as barrier threshold selection and RFE method.
9. Save the best model in the form of a pickle file.
10. Inspect and clean up the data, including data encoding on categorical variables.

---

## Modular Code Overview

1. **input**: Contains all the data files for analysis, such as `Data_regression.csv`.
2. **src**: The most important folder containing all the modularized code for different steps in a modularized manner. This folder consists of `Engine.py` and `ML_Pipeline`.
3. **output**: Contains the best-fitted models trained on the data. These models can be loaded and used for future predictions without the need to retrain them.
4. **lib**: A reference folder containing the original iPython notebook used in the project.

---

## Concepts Explored:

1. Logistic Regression and the logistic function.
2. Coefficients in Logistic Regression.
3. Maximum log-likelihood.
4. Confusion matrix, recall, accuracy, precision, F1-score, AUC, and ROC.
5. Basic Exploratory Data Analysis (EDA)
6. Data inspection and cleaning.
7. Building models with statsmodels and scikit-learn.
8. Model validation with various metrics.
9. Handling unbalanced data with different techniques.
10. Feature selection methods.
11. Saving the best model in pickle format for future use.

---
