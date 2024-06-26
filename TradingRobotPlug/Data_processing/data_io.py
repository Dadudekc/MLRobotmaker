import pandas as pd
import os
import json
from pathlib import Path

def read_csv(file_path):
    """ Read a CSV file and return a DataFrame. """
    return pd.read_csv(file_path)

def write_csv(df, file_path):
    """ Write a DataFrame to a CSV file. """
    df.to_csv(file_path, index=False)

def detect_and_transform_format(file_path):
    """ Detect the format of the file and transform it if necessary. """
    df = pd.read_csv(file_path)

    # Logic for detecting format and transforming
    # (Use the logic from your original script)

    return df

def handle_nan_in_dataframe(df):
    """ Handle NaN values in a DataFrame. """
    # Logic for handling NaN values
    # (Use the logic from your original script)

    return df

def load_transformed_data(file_path):
    """ Load data, transform format, and handle NaN values. """
    df = read_csv(file_path)
    df = detect_and_transform_format(file_path)
    df = handle_nan_in_dataframe(df)
    return df

import os

def save_transformed_data(df, file_path):
    """ Save data to a CSV file. """
    if not os.path.isdir(os.path.dirname(file_path)):
        print("Save directory does not exist.")
        return

    df.to_csv(file_path, index=False)
    print(f"File saved: {file_path}")

