# Import the required libraries
import statsmodels.api as sm
import numpy as np

# Function to create a logistic regression model with the Statsmodels library
def logistic_regression(df, class_col, cols_to_exclude):
    """
    Create a logistic regression model using the Statsmodels library.

    Parameters:
    df (pandas.DataFrame): The dataset.
    class_col (str): The name of the classification variable.
    cols_to exclude (list): A list of columns to exclude from the analysis.

    Returns:
    result (statsmodels.genmod.generalized_linear_model.GLMResultsWrapper): The results of the logistic regression model.
    """
    cols = df.select_dtypes(include=np.number).columns.tolist()
    X = df[cols]
    X = X[X.columns.difference([class_col])]
    X = X[X.columns.difference(cols_to_exclude)]

    y = df[class_col]
    logit_model = sm.Logit(y, X)
    result = logit_model.fit()
    
    # Uncomment the line below to display the summary of the model
    # print(result.summary2())

    return result
