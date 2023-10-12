from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.utils import resample
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.over_sampling import SMOTE

# Function to create a logistic regression model with balanced class weights
def run_model_bweights(X_train, X_test, y_train, y_test):
    """
    Create a logistic regression model with balanced class weights.

    Parameters:
    X_train (array-like): Training data features.
    X_test (array-like): Test data features.
    y_train (array-like): Training data labels.
    y_test (array-like): Test data labels.

    Returns:
    None
    """
    global logreg
    logreg = LogisticRegression(random_state=13, class_weight='balanced')
    logreg.fit(X_train, y_train)
    global y_pred
    y_pred = logreg.predict(X_test)
    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test)
    # Uncomment the following lines to print classification report and AUC
    # print(classification_report(y_test, y_pred))
    # print("The area under the curve is: %0.2f" % logit_roc_auc)

# Function to create a logistic regression model with custom-defined class weights
def run_model_aweights(X_train, X_test, y_train, y_test, w):
    """
    Create a logistic regression model with custom-defined class weights.

    Parameters:
    X_train (array-like): Training data features.
    X_test (array-like): Test data features.
    y_train (array-like): Training data labels.
    y_test (array-like): Test data labels.
    w (dict): Class weights, e.g., {0: 90, 1: 10}.

    Returns:
    None
    """
    global logreg
    logreg = LogisticRegression(random_state=13, class_weight=w)
    logreg.fit(X_train, y_train)
    global y_pred
    y_pred = logreg.predict(X_test)
    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    # Uncomment the following lines to print classification report and AUC
    # print(classification_report(y_test, y_pred))
    # print("The area under the curve is: %0.2f" % logit_roc_auc)

# Function to adjust data imbalance by resampling
def adjust_imbalance(X_train, y_train, class_col):
    """
    Adjust data imbalance by resampling the minority class.

    Parameters:
    X_train (array-like): Training data features.
    y_train (array-like): Training data labels.
    class_col (str): The name of the classification variable.

    Returns:
    resampled_df (pandas.DataFrame): Resampled dataset with balanced classes.
    """
    X = pd.concat([X_train, y_train], axis=1)
    class0 = X[X[class_col] == 0]
    class1 = X[X[class_col] == 1]
    
    if len(class1) < len(class0):
        resampled = resample(class1, replace=True, n_samples=len(class0), random_state=10)
    else:
        resampled = resample(class1, replace=False, n_samples=len(class0), random_state=10)
    
    resampled_df = pd.concat([resampled, class0])
    return resampled_df

# Function to create a logistic regression model with SMOTE
def prepare_model_smote(df, class_col, cols_to_exclude):
    """
    Create a logistic regression model with SMOTE (Synthetic Minority Over-sampling Technique).

    Parameters:
    df (pandas.DataFrame): The dataset.
    class_col (str): The name of the classification variable.
    cols_to_exclude (list): A list of columns to exclude from the analysis.

    Returns:
    X_train, X_test, y_train, y_test (array-like): Split dataset after applying SMOTE.
    """
    cols = df.select_dtypes(include=np.number).columns.tolist()
    X = df[cols]
    X = X[X.columns.difference([class_col])]
    X = X[X.columns.difference(cols_to_exclude)]
    y = df[class_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    sm = SMOTE(random_state=0, sampling_strategy=1.0)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test
