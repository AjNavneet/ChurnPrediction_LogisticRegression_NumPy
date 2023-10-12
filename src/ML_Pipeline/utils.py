import pandas as pd

# Function to read the data file 
def read_data(file_path, **kwargs):
    """
    Read a data file and return it as a DataFrame.

    Parameters:
    file_path (str): The path to the data file.
    **kwargs: Additional keyword arguments to pass to pd.read_csv.

    Returns:
    raw_data (pandas.DataFrame): The raw data from the file.
    """
    raw_data = pd.read_csv(file_path, **kwargs)
    return raw_data

# Function to inspect the dataset
def inspection(dataframe):
    """
    Inspect the dataset and display information about variable types and missing values.

    Parameters:
    dataframe (pandas.DataFrame): The dataset to inspect.
    """
    print("Types of the variables we are working with:")
    print(dataframe.dtypes)

    print("Total Samples with missing values:")
    print(dataframe.isnull().any(axis=1).sum())

    print("Total Missing Values per Variable")
    print(dataframe.isnull().sum())

# Function to remove null values
def null_values(df):
    """
    Remove rows with missing values from the dataset.

    Parameters:
    df (pandas.DataFrame): The dataset.

    Returns:
    df (pandas.DataFrame): The dataset with missing values removed.
    """
    df = df.dropna()
    return df
