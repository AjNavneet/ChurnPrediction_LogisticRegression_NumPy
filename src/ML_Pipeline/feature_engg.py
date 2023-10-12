from sklearn.feature_selection import VarianceThreshold
import numpy as np
from sklearn import preprocessing

class_col = 'churn'
cols_to_exclude = ['customer_id', 'phone_no', 'year']

# Function to select variables with a threshold using VarianceThreshold
def var_threshold_selection(df, cols_to_exclude, class_col, threshold):
    """
    Select numerical variables with a variance above a given threshold.

    Parameters:
    df (pandas.DataFrame): The dataset.
    cols_to_exclude (list): A list of columns to exclude from the analysis.
    class_col (str): The name of the classification variable.
    threshold (float): The threshold for variance selection.

    Returns:
    None
    """
    cols = df.select_dtypes(include=np.number).columns.tolist()
    X = df[cols]
    X = X[X.columns.difference(cols_to_exclude)]
    X = X[X.columns.difference([class_col])]

    # Scaling variables
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    var_thr = VarianceThreshold(threshold=threshold)
    var_thr.fit(X_scaled)
    selected_cols = X.columns[var_thr.get_support()]

    print("The selected features are:")
    print(list(selected_cols))

# Function to select variables using Recursive Feature Elimination (RFE)
def rfe_selection(df, cols_to_exclude, class_col, model):
    """
    Perform feature selection using Recursive Feature Elimination (RFE) with a specified model.

    Parameters:
    df (pandas.DataFrame): The dataset.
    cols_to_exclude (list): A list of columns to exclude from the analysis.
    class_col (str): The name of the classification variable.
    model: The machine learning model used for feature selection.

    Returns:
    None
    """
    import warnings
    warnings.filterwarnings("ignore")
    cols = df.select_dtypes(include=np.number).columns.tolist()
    X = df[cols]
    X = X[X.columns.difference(cols_to_exclude)]
    X = X[X.columns.difference([class_col])]
    y = df[class_col]

    rfe = RFE(model)
    rfe = rfe.fit(X, y)
    global selected_cols
    selected_cols = X.columns[rfe.support()]

    print("The selected features are:")
    print(list(selected_cols))
