from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

# Function to prepare the dataset for the machine learning model
def prepare_model(df, class_col, cols_to_exclude):
    """
    Prepare the dataset for a machine learning model.

    Parameters:
    df (pandas.DataFrame): The dataset.
    class_col (str): The name of the classification variable.
    cols_to_exclude (list): A list of columns to exclude from the analysis.

    Returns:
    X_train, X_test, y_train, y_test (array-like): Split dataset for training and testing.
    """
    ## Selecting only the numerical columns and excluding the specified columns
    cols = df.select_dtypes(include=np.number).columns.tolist()
    X = df[cols]
    X = X[X.columns.difference([class_col])]
    X = X[X.columns.difference(cols_to_exclude)]
    
    ## Selecting y as a column
    y = df[class_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    return X_train, X_test, y_train, y_test

# Function to run the machine learning model
def run_model(X_train, X_test, y_train, y_test):
    """
    Run a machine learning model using logistic regression.

    Parameters:
    X_train (array-like): Training data features.
    X_test (array-like): Test data features.
    y_train (array-like): Training data labels.
    y_test (array-like): Test data labels.

    Returns:
    logreg (LogisticRegression): The trained logistic regression model.
    y_pred (array-like): Predicted labels.
    """
    ## Fitting the logistic regression
    logreg = LogisticRegression(random_state=13)
    logreg.fit(X_train, y_train)
    
    ## Predicting y values
    y_pred = logreg.predict(X_test)
    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    
    # Printing classification report and AUC score
    print(classification_report(y_test, y_pred))
    print("The area under the curve is: %0.2f" % logit_roc_auc)
    
    return logreg, y_pred
