#data_transformations.py

import pandas as pd
import pandas as pd
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_columns(df):
    """
    Standardize column names to a common format.

    Args:
        df (pd.DataFrame): The DataFrame whose columns need to be normalized.

    Returns:
        pd.DataFrame: DataFrame with normalized column names.
    """
    # Example: Convert column names to lower case and replace spaces with underscores
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def convert_to_datetime(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Convert a column in the DataFrame to datetime.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        date_column (str): The name of the column to convert to datetime.

    Returns:
        pd.DataFrame: The DataFrame with the specified column converted to datetime.

    Raises:
        ValueError: If the specified date column is not found in the DataFrame.
    """
    try:
        if date_column not in df.columns:
            raise ValueError(f"Column '{date_column}' not found in the DataFrame.")
        df[date_column] = pd.to_datetime(df[date_column])
    except Exception as e:
        logging.error(f"Error in converting column to datetime: {e}")
        raise
    return df

def fill_missing_values(df, method='ffill'):
    """
    Fill missing values in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        method (str): The method to use for filling missing values ('ffill', 'bfill', etc.)

    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    df.fillna(method=method, inplace=True)
    return df

def standardize_data(df):
    """
    Apply a standard set of transformations to clean and standardize the data.

    Args:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: The standardized DataFrame.
    """
    df = normalize_columns(df)
    # Assuming 'date' is the column to be converted to datetime
    df = convert_to_datetime(df, 'date')
    df = fill_missing_values(df)
    return df

def aggregate_data(df, freq='M'):
    """
    Aggregate data to a specified frequency.

    Args:
        df (pd.DataFrame): The DataFrame to aggregate.
        freq (str): The frequency for aggregation ('M' for monthly, 'W' for weekly, etc.)

    Returns:
        pd.DataFrame: Aggregated DataFrame.
    """
    df = df.set_index('date')
    aggregated_df = df.resample(freq).mean()
    return aggregated_df.reset_index()
