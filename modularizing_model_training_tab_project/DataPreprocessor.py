import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Class for preprocessing data by incorporating feature engineering, data cleaning, and splitting functionalities.
    """
    def __init__(self, lag_sizes=None, window_sizes=None):
        """
        Initializes the DataPreprocessor with optional lag and window sizes for feature engineering.
        
        Args:
            lag_sizes (list of int): Default lag sizes for creating lag features if none are provided.
            window_sizes (list of int): Default window sizes for creating rolling window features if none are provided.
        """
        self.lag_sizes = lag_sizes or [1, 2, 3, 5, 10]
        self.window_sizes = window_sizes or [5, 10, 20]

    def preprocess_data(self, data, target_column='close'):
        """
        Main method to preprocess data which includes cleaning, feature engineering and missing data handling.
        
        Args:
            data (pd.DataFrame): The input dataset.
            target_column (str): Target column for lag and rolling features.
        
        Returns:
            pd.DataFrame: The preprocessed data ready for modeling.
        """
        if data.empty:
            logger.error("Input dataset is empty.")
            return None

        # Data cleaning and feature engineering
        data = self.clean_column_names(data)
        data = self.add_date_features(data)
        data = self.add_lag_features(data, target_column)
        data = self.add_rolling_window_features(data, target_column)
        data.dropna(inplace=True)

        if data.empty:
            logger.error("Dataset is empty after preprocessing. Check feature engineering parameters.")
            return None

        return data

    def clean_column_names(self, data):
        """
        Cleans column names by removing extra spaces and replacing special characters.
        """
        data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^a-z0-9_]', '')
        return data

    def add_date_features(self, data):
        """
        Adds date-related features (day of week, month, year) if a 'date' column exists.
        """
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data['day_of_week'] = data['date'].dt.dayofweek
            data['month'] = data['date'].dt.month
            data['year'] = data['date'].dt.year
        return data

    def add_lag_features(self, data, target_column):
        """
        Generates lag features based on specified lag sizes.
        """
        for lag in self.lag_sizes:
            data[f'{target_column}_lag_{lag}'] = data[target_column].shift(lag)
        return data

    def add_rolling_window_features(self, data, target_column):
        """
        Generates rolling window statistics based on specified window sizes.
        """
        for window in self.window_sizes:
            data[f'{target_column}_rolling_mean_{window}'] = data[target_column].rolling(window).mean()
            data[f'{target_column}_rolling_std_{window}'] = data[target_column].rolling(window).std()
        return data

    def split_data(self, data, test_size=0.2, random_state=None):
        """
        Splits the data into training and testing sets.
        
        Args:
            data (pd.DataFrame): The dataset to split.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Seed used by the random number generator.
        
        Returns:
            tuple: (train_data, test_data)
        """
        if 'target' not in data.columns:
            logger.error("Target column not found in dataset for splitting.")
            return None, None

        X = data.drop('target', axis=1)
        y = data['target']
        train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logger.info("Data has been split into training and testing sets.")
        return (train_data, test_labels), (test_data, test_labels)

# Usage
# preprocessor = DataPreprocessor()
# data = pd.read_csv('path/to/your/data.csv')
# processed_data = preprocessor.preprocess_data(data)
# train, test = preprocessor.split_data(processed_data)
