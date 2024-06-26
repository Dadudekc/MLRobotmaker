#data_preprocessing.py

import pandas as pd
import numpy as np
import logging

def preprocess_data(data, fill_method='ffill', date_column=None, target_column=None):
    """
    Perform preprocessing on the given dataset.

    Args:
        data (DataFrame): The dataset to preprocess.
        fill_method (str): The method used to fill missing values.
        date_column (str): The name of the date column to standardize, if present.
        target_column (str): The name of the target column to standardize, if present.

    Returns:
        DataFrame: The preprocessed dataset.
    """
    if date_column:
        data = standardize_date_column(data, date_column)
    data = fill_missing_values(data, fill_method)
    if target_column:
        data = standardize_target_column(data, target_column)
    return data

def standardize_date_column(data, date_column):
    """
    Standardize the date column in the dataset.

    Args:
        data (DataFrame): The dataset to process.
        date_column (str): The name of the date column to standardize.

    Returns:
        DataFrame: Dataset with standardized date column.
    """
    if date_column in data.columns:
        data[date_column] = pd.to_datetime(data[date_column])
        data.sort_values(by=date_column, inplace=True)
        return data
    else:
        logging.error(f"Date column '{date_column}' not found in data.")
        raise ValueError(f"Date column '{date_column}' not found in data.")

def fill_missing_values(data, fill_method):
    """
    Fill missing values in the dataset using the specified method.

    Args:
        data (DataFrame): The dataset to process.
        fill_method (str): Method used to fill missing values.

    Returns:
        DataFrame: Dataset with missing values filled.
    """
    valid_methods = ['ffill', 'bfill', 'mean', 'median']
    if fill_method not in valid_methods:
        logging.error(f"Invalid fill method: {fill_method}. Valid methods are: {valid_methods}")
        raise ValueError(f"Invalid fill method: {fill_method}. Valid methods are: {valid_methods}")

    if fill_method in ['ffill', 'bfill']:
        data.fillna(method=fill_method, inplace=True)
    else:
        data.fillna(data.agg(fill_method), inplace=True)
    return data

def standardize_target_column(data, target_column):
    """
    Rename the target column in the dataset to a standard name.

    Args:
        data (DataFrame): The dataset to process.
        target_column (str): The current name of the target column.

    Returns:
        DataFrame: Dataset with the target column renamed.
    """
    if target_column in data.columns:
        data.rename(columns={target_column: 'target'}, inplace=True)
        return data
    else:
        logging.error(f"Target column '{target_column}' not found in data.")
        raise ValueError(f"Target column '{target_column}' not found in data.")

# Additional functions like 'get_features_from_csv', 'convert_and_sort_by_date' can also be added here