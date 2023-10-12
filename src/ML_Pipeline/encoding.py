from sklearn.preprocessing import OrdinalEncoder

def encode_categories(df, variables):
    """
    Encode categorical variables in the DataFrame using OrdinalEncoder.

    Parameters:
    df (pandas.DataFrame): The dataset to encode.
    variables (list): A list of column names containing categorical variables.

    Returns:
    None
    """
    ord_enc = OrdinalEncoder()
    for v in variables:
        name = v + '_code'
        df[name] = ord_enc.fit_transform(df[[v]])
        # Uncomment the following lines to print unique encoded values
        # print('The encoded values for ' + v + ' are:')
        # print(df[name].unique())
