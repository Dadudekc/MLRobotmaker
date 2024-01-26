# model_training.py

# Section 1: Imports and Setup
import logging
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import torch
import sklearn.base
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import save_model as save_keras_model
from tensorflow.keras.models import load_model as load_keras_model
from kerastuner.engine.hypermodel import HyperModel
from kerastuner.tuners import BayesianOptimization
from statsmodels.tsa.arima.model import ARIMA
import statsmodels
import joblib  # For scikit-learn models
import tensorflow as tf  # For TensorFlow/Keras models
import xgboost as xgb  # For XGBoost models



# Section 2: Model Training Functions - Part 1

def train_model(X_train, y_train, model_type='linear_regression'):
    """
    Train a machine learning model based on the specified type.

    Args:
        X_train (DataFrame): Training features.
        y_train (Series): Training target.
        model_type (str): Type of model to train.

    Returns:
        model: Trained model object.
    """
    if model_type == 'linear_regression':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor()
    elif model_type == 'neural_network':
        model = create_neural_network(X_train.shape[1])
    elif model_type == 'LSTM':
        model = create_lstm_model(X_train.shape[1:])
    elif model_type == 'ARIMA':
        model = train_arima_model(X_train, y_train)
    else:
        logging.error(f"Unsupported model type: {model_type}")
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train, y_train)
    return model

# Function for creating a customizable neural network model


def create_neural_network(input_shape, layers=[128, 64], output_units=1, output_activation=None,
                          activation='relu', dropout_rate=0.0, optimizer='adam', loss='mean_squared_error', metrics=None):
    """
    Create a customizable neural network for regression tasks with extended flexibility.

    Args:
        input_shape (tuple): The shape of the input data, excluding the batch size.
        layers (list): List of integers, the size of each dense layer.
        output_units (int): Number of units in the output layer.
        output_activation (str): Activation function for the output layer.
        activation (str): Activation function for the hidden layers.
        dropout_rate (float): Dropout rate, between 0 and 1.
        optimizer (str): Optimizer to use.
        loss (str): Loss function.
        metrics (list): List of metrics to be evaluated by the model during training and testing.

    Returns:
        tf.keras.Model: A compiled neural network model.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))  # Use the tuple input_shape directly

    for units in layers:
        model.add(tf.keras.layers.Dense(units, activation=activation))
        if dropout_rate > 0.0:
            model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(output_units, activation=output_activation))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


# Function for creating an LSTM model (Placeholder - Implement as needed)
def create_lstm_model(input_shape, lstm_layers=[50, 50], dropout_rates=[0.2, 0.2], optimizer='adam', loss='mean_squared_error'):
    """
    Create an advanced LSTM model with customizable layers and dropout rates.

    Args:
        input_shape (tuple): The shape of the input data.
        lstm_layers (list): List of LSTM units for each layer.
        dropout_rates (list): List of dropout rates corresponding to each LSTM layer.
        optimizer (str or keras.optimizers): Optimizer to use.
        loss (str or callable): Loss function to use.

    Returns:
        model: A compiled LSTM model.
    """
    model = tf.keras.Sequential()

    for i, (units, dropout_rate) in enumerate(zip(lstm_layers, dropout_rates)):
        if i == 0:
            model.add(tf.keras.layers.LSTM(units, input_shape=input_shape, return_sequences=(i < len(lstm_layers) - 1)))
        else:
            model.add(tf.keras.layers.LSTM(units, return_sequences=(i < len(lstm_layers) - 1)))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(1))  # Output layer - adjust based on your needs
    model.compile(optimizer=optimizer, loss=loss)
    return model


# Function to train an ARIMA model (Placeholder - Implement as needed)
def train_arima_model(X_train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), trend=None):
    """
    Train an enhanced ARIMA model, including support for seasonal components.

    Args:
        X_train (array-like): Training features.
        order (tuple): The (p, d, q) order of the ARIMA model.
        seasonal_order (tuple): The (P, D, Q, S) seasonal order of the ARIMA model.
        trend (str or None): The trend parameter.

    Returns:
        model_fit: A fitted ARIMA model.
    """
    model = ARIMA(X_train, order=order, seasonal_order=seasonal_order, trend=trend)
    model_fit = model.fit()
    return model_fit


# Function for creating an ensemble model (Placeholder - Implement as needed)
def create_ensemble_model(models, weights=None):

    """
    Create a sophisticated ensemble model, optionally using weights for averaging.

    Args:
        models (list): List of trained models.
        weights (list or None): List of weights for each model.

    Returns:
        ensemble_model: An ensemble model.
    """
    class SophisticatedEnsembleModel:
        def __init__(self, models, weights):
            self.models = models
            self.weights = weights if weights else np.ones(len(models)) / len(models)

        def predict(self, X):
            weighted_predictions = np.sum([w * model.predict(X) for w, model in zip(self.weights, self.models)], axis=0)
            return weighted_predictions / np.sum(self.weights)

    return SophisticatedEnsembleModel(models, weights)



# Section 3: Model Evaluation

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.

    Args:
        model: The trained model to evaluate.
        X_test (DataFrame): Test features.
        y_test (Series): Test target.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return {'mean_squared_error': mse, 'r2_score': r2}

# Section 4: Model Saving

def save_model(model, filename):
    """
    Save the trained model to a file in an appropriate format based on its type.

    Args:
        model: The trained model to save.
        filename (str): File path to save the model.

    Returns:
        None
    """
    if isinstance(model, sklearn.base.BaseEstimator):
        joblib.dump(model, filename + '.joblib')
    elif isinstance(model, keras.Model):
        save_keras_model(model, filename + '.h5')
    elif isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), filename + '.pth')
    else:
        raise ValueError("Model type not supported")

    logging.info(f"Model saved to {filename}")

# Section 5: Hyperparameter Tuning

class CustomHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        """
        Build a neural network model with hyperparameters specified by Keras Tuner.

        Args:
            hp: Hyperparameters provided by Keras Tuner.

        Returns:
            model: A neural network model with specified hyperparameters.
        """
        model = keras.Sequential()

        # Define the number of units in the first dense layer as a hyperparameter
        units = hp.Int('units', min_value=32, max_value=512, step=32)
        model.add(keras.layers.Dense(units=units, activation='relu', input_shape=(self.input_shape,)))

        # Optionally add more dense layers
        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(keras.layers.Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32), activation='relu'))
            if hp.Choice('dropout', [True, False]):
                model.add(keras.layers.Dropout(rate=hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)))
            model.add(keras.layers.BatchNormalization())

        # Define the learning rate for the optimizer as a hyperparameter
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.add(keras.layers.Dense(1, activation='linear'))  # Output layer for regression
        
        # Compile the model with the defined optimizer and loss function
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        return model

def perform_hyperparameter_tuning(X_train, y_train, model_type='neural_network', input_shape=None, max_trials=10, epochs=20):
    """
    Perform advanced hyperparameter tuning using Bayesian optimization or appropriate methods.

    Args:
        X_train (DataFrame): Training features.
        y_train (Series): Training target.
        model_type (str): Type of model to train ('neural_network', 'random_forest', etc.).
        input_shape (tuple): Input shape for neural network models.
        max_trials (int): Maximum number of trials for hyperparameter search.
        epochs (int): Number of epochs for training neural network models.

    Returns:
        model: The best model found through hyperparameter tuning.
    """
    if model_type == 'neural_network':
        assert input_shape is not None, "Input shape must be provided for neural network models."

        hypermodel = CustomHyperModel(input_shape=input_shape)

        tuner = BayesianOptimization(
            hypermodel,
            objective='val_loss',
            max_trials=max_trials,
            directory='my_neural_network_tuner_dir',
            project_name='neural_network_tuning'
        )

        tuner.search(X_train, y_train, epochs=epochs, validation_split=0.2)
        best_model = tuner.get_best_models(num_models=1)[0]

    elif model_type == 'random_forest':
        # Define parameter space for Random Forest
        param_distributions = {
            # Add random forest hyperparameters here
        }
        
        model = RandomForestRegressor()
        tuner = RandomizedSearchCV(
            model,
            param_distributions=param_distributions,
            n_iter=max_trials,
            cv=5,
            n_jobs=-1,
            verbose=1
        )

        tuner.fit(X_train, y_train)
        best_model = tuner.best_estimator_

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return best_model

def load_model(file_path):
    """
    Load a model from a specified file path.

    Args:
        file_path (str): The path to the model file.

    Returns:
        model: The loaded model.
    """
    # Determine the model type based on the file extension
    if file_path.endswith('.joblib'):
        # Load a scikit-learn model
        return joblib.load(file_path)
    elif file_path.endswith('.h5'):
        # Load a Keras model
        return tf.keras.models.load_model(file_path)
    elif file_path.endswith('.json'):
        # Load a XGBoost model saved as JSON
        model = xgb.XGBModel()
        model.load_model(file_path)
        return model
    else:
        raise ValueError("Unsupported model file format or extension")



def train_hist_gradient_boosting(self, data_file_path, scaler_type, target_column='close'):
    try:
        # Load the data from the provided file path (modify this as needed)
        df = pd.read_csv(data_file_path)

        # Handle missing values (you can choose forward fill or backward fill)
        df.fillna(method='ffill', inplace=True)  # Use forward fill for missing values

        # Extract features and target
        X = df.drop(target_column, axis=1)  # Assuming the target column is labeled 'close'
        y = df[target_column]

        # Continue with the rest of your code (scaling, splitting, and training)
        # Perform data scaling if needed (you can choose StandardScaler or other scalers)
        if scaler_type == "standard":
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")

        X_scaled = scaler.fit_transform(X)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Create an instance of the GradientBoostingRegressor
        gbr = GradientBoostingRegressor()

        # Train the model
        gbr.fit(X_train, y_train)

        # Evaluate the model (you can add evaluation metrics here)
        train_score = gbr.score(X_train, y_train)
        test_score = gbr.score(X_test, y_test)

        # You can print or log the training and testing scores
        print(f"Training Score: {train_score}")
        print(f"Testing Score: {test_score}")

        # Return the trained model
        return gbr

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
# Example usage
# model = load_model('path_to_your_model_file')
