#model_development.py

"""
This script aims to preprocess data and train machine learning models.
It reads configurations for working directories and other parameters from a configuration file.
"""

# Section 1: Imports and Initial Setup
import os
import logging
import configparser
import pandas as pd
import numpy as np
import joblib
import json
import shap
import tensorflow as tf

from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_tuner import HyperModel, RandomSearch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, RobustScaler, QuantileTransformer, Normalizer, MaxAbsScaler
from keras.layers import LSTM
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import precision_score, recall_score, f1_score


# Setup Logging
logging.basicConfig(level=logging.INFO)

# Initialize global variables for best model and MSE
best_mse = float('inf')
best_model = None

# Function to load the configuration file
def load_configuration(config_file='config.ini'):
    config = configparser.ConfigParser()
    if not os.path.exists(config_file):
        logging.error(f"Configuration file does not exist: {config_file}")
        raise FileNotFoundError(f"Configuration file does not exist: {config_file}")
    config.read(config_file)
    logging.info("Configuration file loaded and validated successfully.")
    return config

# Function to read CSV file and return features list
def get_features_from_csv(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path)
        return df.columns.tolist()
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        return []

# Function for interactive feature selection
def select_features_interactively(features_list):
    print("Please select the features you want to include in the model training:")
    print("0. Finish selection")
    print("ALL. Select all features")
    selected_features = []
    for i, feature in enumerate(features_list, 1):
        print(f"{i}. {feature}")

    while True:
        choice = input("Enter the number of the feature to toggle selection (0 to finish, ALL for all): ")
        if choice == '0':
            break
        elif choice == 'ALL':
            selected_features = features_list.copy()
            break
        elif choice.isdigit() and int(choice) <= len(features_list):
            selected_feature = features_list[int(choice) - 1]
            if selected_feature not in selected_features:
                selected_features.append(selected_feature)
            else:
                selected_features.remove(selected_feature)
        else:
            print("Invalid input. Please enter a valid number.")
    return selected_features

# Load configuration file
config = load_configuration('C:\\Users\\Dagurlkc\\OneDrive\\Desktop\\DaDudeKC\\MLRobot\\config.ini')

# Directory containing the CSV files
directory_path = config['Paths']['data_folder']

def validate_section_keys(config, section, required_keys):
    """
    Validate that all necessary keys are present in the specified configuration section.
    
    Args:
        config (ConfigParser object): Configuration object to validate.
        section (str): Section name to check in the configuration.
        required_keys (list): Required keys to check for in the section.
    
    Raises:
        ValueError: If any required keys are missing.
    """
    for key in required_keys:
        if key not in config[section]:
            raise ValueError(f"Missing required config key in {section}: {key}")

# Section 3: Data Preprocessing
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
    log_training_message("Starting preprocessing...", log_text)

    # Verify and standardize the date column if specified
    if date_column and date_column in data.columns:
        # Convert to datetime and handle NaN values
        data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
        data.dropna(subset=[date_column], inplace=True)

        # Debug: Check the data types after conversion
        print("Data types after date conversion:", data.dtypes)

        # Extract datetime features and convert to timestamp
        data['year'] = data[date_column].dt.year
        data['month'] = data[date_column].dt.month
        data['day'] = data[date_column].dt.day
        data['dayofweek'] = data[date_column].dt.dayofweek
        data['timestamp'] = data[date_column].astype(int)

        # Debug: Check the DataFrame after all conversions
        print("Data after date processing:", data.head())

        data.drop(date_column, axis=1, inplace=True)

        # Optionally, convert to a timestamp (numeric format)
        data['timestamp'] = data[date_column].astype(int)

        # Drop original date column
        data.drop(date_column, axis=1, inplace=True)

        log_training_message(f"Data after sorting by date and feature extraction:\n{data.head()}", log_text)

    # Fill missing values
    data.fillna(method=fill_method, inplace=True)
    log_training_message("Missing values filled.", log_text)

    # Standardize the naming of the target column if specified
    if target_column and target_column in data.columns:
        data.rename(columns={target_column: 'target'}, inplace=True)
        log_training_message(f"Renamed target column to 'target'.", log_text)

    log_training_message("Preprocessing completed.", log_text)
    
    return data


def fill_missing_values(data, fill_method):
    """
    Fill missing values in the dataset using the specified method.

    Args:
        data (DataFrame): The dataset to process.
        fill_method (str): Method used to fill missing values.

    Returns:
        DataFrame: Dataset with missing values filled.
    """
    if fill_method not in ['ffill', 'bfill', 'mean', 'median']:
        raise ValueError(f"Invalid fill method: {fill_method}")
    
    data.fillna(method=fill_method, inplace=True)
    return data

def standardize_target_column(data, target_column):
    """
    Standardize the naming of the target column.

    Args:
        data (DataFrame): The dataset to process.
        target_column (str): The target column name.

    Returns:
        DataFrame: Dataset with standardized target column name.
    """
    data.rename(columns={target_column: 'target'}, inplace=True)
    return data


def standardize_date_column(data):
    """
    Checks for the presence of a date column and standardizes it to 'date'.

    Args:
        data (DataFrame): The dataset to check.

    Returns:
        str: The name of the date column.

    Raises:
        ValueError: If a date column is not found.
    """
    date_column_candidates = ['date', 't']
    for candidate in date_column_candidates:
        if candidate in data.columns:
            data.rename(columns={candidate: 'date'}, inplace=True)
            return 'date'
    raise ValueError("'date' or 't' column missing in data")

def convert_and_sort_by_date(data, date_column):
    """
    Converts the date column to datetime objects and sorts the dataframe by date.

    Args:
        data (DataFrame): The dataset to process.
        date_column (str): The name of the date column.

    Returns:
        DataFrame: Dataset with converted and sorted date column.

    Raises:
        ValueError: If date conversion fails.
    """
    try:
        data[date_column] = pd.to_datetime(data[date_column])
        data.sort_values(by=date_column, inplace=True)
    except Exception as e:
        raise ValueError(f"Date conversion failed: {e}")
    
    return data

# Section 4: Data Splitting and Scaling

def split_and_scale_data(X, y, test_size, scaler_type):
    # Define the valid scalers
    valid_scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler(),
        'quantile': QuantileTransformer(),
        'power': PowerTransformer(),
        'normalizer': Normalizer(),
        'maxabs': MaxAbsScaler()
    }

    # Make sure the provided scaler_type is valid
    if scaler_type not in valid_scalers:
        raise ValueError(f"Invalid scaler type '{scaler_type}'. Valid options are: {list(valid_scalers.keys())}")

    # Exclude non-numeric columns for scaling
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X_numeric = X[numeric_columns]

    # Convert test_size to float
    test_size = float(test_size)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=test_size, random_state=42)
    logging.info(f"Data split into training and testing sets with test size = {test_size}")

    # Scale the data
    scaler = valid_scalers[scaler_type]
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logging.info(f"Data scaled using '{scaler_type}' scaler")

    return X_train_scaled, X_test_scaled, y_train, y_test



# Section 5: Custom HyperModel Class for Keras Tuner

class CustomHyperModel(HyperModel):
    """
    Custom HyperModel for Keras Tuner.
    Defines a space of hyperparameters and builds a model for each set of hyperparameters.
    """

    def __init__(self, input_shape):
        """
        Initialize the HyperModel with the shape of the input data.

        Args:
            input_shape (tuple): The shape of the input data.
        """
        self.input_shape = input_shape

def build(self, hp):
    """
    Build a model with the given hyperparameters.

    Args:
        hp (HyperParameters): The hyperparameters to build the model with.

    Returns:
        Model: A compiled Keras model.
    """
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])


    # Adding more hidden layers with flexible unit counts and activations
    for i in range(hp.Int('num_hidden_layers', 1, 3)):
        model.add(Dense(
            units=hp.Int(f'units_layer_{i}', min_value=16, max_value=128, step=16),
            activation=hp.Choice(f'activation_layer_{i}', ['relu', 'tanh', 'sigmoid'])
        ))

    model.add(Dense(1, activation='linear'))  # Output layer

    model.compile(
        optimizer=hp.Choice('optimizer', ['adam', 'sgd']),
        loss='mse',
        metrics=['mean_squared_error']
    )
    return model





#hyperparameter tuning function

def bayesian_hyperparameter_tuning(X_train, y_train, model, search_space, n_iter=50, cv=3):
    """
    Perform hyperparameter tuning using Bayesian Optimization.

    Args:
        X_train (ndarray): Training data.
        y_train (ndarray): Labels for training data.
        model (Model): The machine learning model to tune.
        search_space (dict): The hyperparameter search space.
        n_iter (int): Number of iterations for the search.
        cv (int): Number of folds for cross-validation.

    Returns:
        Model: The best model found by the Bayesian optimizer.
    """
    bayes_cv_tuner = BayesSearchCV(
        estimator=model,
        search_spaces=search_space,
        scoring='neg_mean_squared_error',
        n_iter=n_iter,
        cv=cv,
        verbose=1,
        n_jobs=-1
    )

    bayes_cv_tuner.fit(X_train, y_train)
    best_model = bayes_cv_tuner.best_estimator_
    return best_model



# Section 6: Model Training and Evaluation

def evaluate_model(model, X_test, y_test, task_type='regression'):
    """
    Evaluate a given model using the test data and specified metrics.

    Args:
        model (Model): The trained model to evaluate.
        X_test (ndarray): Test features.
        y_test (ndarray): True labels for the test data.
        task_type (str): Type of task ('regression' or 'classification').

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    predictions = model.predict(X_test)
    evaluation_results = {}

    if task_type == 'regression':
        evaluation_results['mean_squared_error'] = mean_squared_error(y_test, predictions)
        evaluation_results['r2_score'] = r2_score(y_test, predictions)
        # Add other regression metrics here
    elif task_type == 'classification':
        # Convert predictions to discrete values if needed
        predictions = np.round(predictions)  # or use an appropriate threshold
        evaluation_results['precision'] = precision_score(y_test, predictions, average='binary')
        evaluation_results['recall'] = recall_score(y_test, predictions, average='binary')
        evaluation_results['f1_score'] = f1_score(y_test, predictions, average='binary')
        # Add other classification metrics here

    return evaluation_results

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type, epochs):
    """
    Train and evaluate a machine learning model.

    Args:
        X_train (ndarray): Training features.
        y_train (ndarray): Training labels.
        X_test (ndarray): Test features.
        y_test (ndarray): Test labels.
        model_type (str): The type of model to train.
        epochs (int): The number of epochs for training.

    Returns:
        Model: The trained model.
        dict: A dictionary containing evaluation metrics for the model.
    """
    logging.info("Starting model training...")

    # Model creation based on the model_type
    if model_type == 'linear_regression':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor()
    elif model_type == 'neural_network':
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
    else:
        raise ValueError("Unsupported model type")

    # Train the model
    if model_type == 'neural_network':
        # For neural network, use callbacks and validation split
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
        model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])
    else:
        # For other models
        model.fit(X_train, y_train)

    # Evaluate the model
    performance = evaluate_model(model, X_test, y_test)

    return model, performance


def save_model_with_metadata(model, model_path, metadata):
    """
    Save a trained model and its metadata to the specified path.

    Args:
        model (Model): The trained model to save.
        model_path (str): The file path to save the model to.
        metadata (dict): The metadata associated with the model.
    """
    # Ensure the save directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save the model
    joblib.dump(model, model_path)

    # Save the metadata
    metadata_path = os.path.splitext(model_path)[0] + '_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

    logging.info(f"Model and metadata saved at {model_path} and {metadata_path}.")

def select_best_model_and_save(models, models_directory, X_test, y_test):
    """
    Evaluate a list of models, select the best one, and save it.

    Args:
        models (list): A list of trained models to evaluate.
        models_directory (str): The directory where the best model will be saved.
        X_test (ndarray): Test features.
        y_test (ndarray): Test labels.

    Returns:
        Model: The best model according to the evaluation.
    """
    best_model = None
    best_performance = {'mean_squared_error': float('inf')}

    for i, model in enumerate(models):
        _, performance = train_and_evaluate_model(model, X_test, y_test)

        if performance['mean_squared_error'] < best_performance['mean_squared_error']:
            best_model = model
            best_performance = performance
            model_filename = f"best_model_{i}.pkl"
            save_model_with_metadata(model, os.path.join(models_directory, model_filename), performance)

    return best_model

def calculate_directional_accuracy(y_true, y_pred):
    # Assuming y_true and y_pred are both numpy arrays or similar sequences
    correct_direction_count = np.sum(np.sign(y_pred[1:] - y_pred[:-1]) == np.sign(y_true[1:] - y_true[:-1]))
    directional_accuracy = correct_direction_count / (len(y_true) - 1)
    return directional_accuracy

def process_file(file_path, selected_features, config, model_type):
    logging.info(f"Processing file: {file_path}")
    logging.info(f"Selected features: {selected_features}")
    logging.info(f"Model type selected: {model_type}")

    data = pd.read_csv(file_path)

    # Ensure selected features are in the current file
    available_features = data.columns.intersection(selected_features)
    X = data[available_features]
    y = data[config['Model']['target_column']]

    logging.info(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

    # Split and scale data
    X_train, X_test, y_train, y_test = split_and_scale_data(X, y, test_size=float(config['Training']['test_size']), scaler_type=config['Training']['scaler_type'])

    # Train the model
    model = train_model(X_train, y_train, model_type)

    # Evaluate the model
    performance = evaluate_model(model, X_test, y_test, task_type='regression')

    # Save the model and its performance metrics
    model_filename = os.path.join(config['Paths']['models_directory'], f"model_{os.path.basename(file_path)}.pkl")
    save_model_with_metadata(model, model_filename, performance)

    # Optionally, print the performance metrics
    print(f"Performance for model trained on {file_path}: {performance}")

    return model, X_train  # Returning the model and training data

def perform_shap_analysis(model, model_type, X_train):
    logging.info("Performing SHAP analysis...")
    print("Shape of X_train:", X_train.shape)
    print("Type of X_train:", type(X_train))

    # Ensuring X_train is in a suitable format for SHAP (e.g., NumPy array)
    if not isinstance(X_train, np.ndarray):
        X_train = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)

    if model_type == 'random_forest':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
    elif model_type == 'linear_regression':
        explainer = shap.KernelExplainer(model.predict, X_train)
        shap_values = explainer.shap_values(X_train)
    elif model_type == 'neural_network' or model_type == 'lstm':
        # Use DeepExplainer for neural network and LSTM models
        if not isinstance(model, tf.keras.Model):
            model = tf.keras.models.model_from_json(model.to_json())
        explainer = shap.DeepExplainer(model, X_train)
        shap_values = explainer.shap_values(X_train)
    else:
        raise ValueError("Unsupported model type for SHAP")

    return shap_values


def select_model_type():
    model_types = ['linear_regression', 'random_forest', 'neural_network']
    print("Select the model type for training:")
    for i, model_type in enumerate(model_types, 1):
        print(f"{i}. {model_type}")

    while True:
        choice = input("Enter the number of the model type: ")
        if choice.isdigit() and 1 <= int(choice) <= len(model_types):
            return model_types[int(choice) - 1]
        else:
            print("Invalid input. Please enter a valid number.")

# Add this function to the existing code

def compare_models(models, X_test, y_test, metrics=None):
    """
    Compare a list of models on the same test set and print their performance.

    Args:
        models (list): A list of models to compare.
        X_test (ndarray): Test features.
        y_test (ndarray): Test labels.
        metrics (dict): A dictionary of metrics to compute.
    """
    if metrics is None:
        metrics = {
            'mean_squared_error': mean_squared_error,
            'r2_score': r2_score
            # Add more metrics here if needed
        }

    for model in models:
        print(f"Evaluating model: {model.__class__.__name__}")
        performance = evaluate_model(model, X_test, y_test, metrics)
        print(f"Performance: {performance}\n")


def main():
    config = load_configuration()  # Load user configurations

    # Directory containing the CSV files
    data_folder = config['Paths']['data_folder']

    # Reading the first file from the directory to get the features
    first_file_path = os.path.join(data_folder, os.listdir(data_folder)[0])
    features_list = get_features_from_csv(first_file_path)

    # Select features interactively based on the first file
    selected_features = select_features_interactively(features_list)

    # Select model type interactively
    model_type = select_model_type()
    print(f"Selected Model Type: {model_type}")

    # Process each file in the data folder
    for csv_file in os.listdir(data_folder):
        if csv_file.endswith('.csv'):
            file_path = os.path.join(data_folder, csv_file)
            print(f"Processing file: {file_path}")

            # Process the file and get the model and training data
            model, X_train = process_file(file_path, selected_features, config, model_type)

            # Perform SHAP analysis
            print(f"Performing SHAP analysis for {model_type} model on {file_path}")
            shap_values = perform_shap_analysis(model, model_type, X_train)

            # Handle SHAP values here (e.g., visualize, save, etc.)

            # Assuming `model` is your trained model and `X_train` is your training data
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_train)

            # Plotting the summary plot
            shap.summary_plot(shap_values, X_train, plot_type="bar")

if __name__ == "__main__":
    main()

