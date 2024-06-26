# model_training.py

# Section 1: Imports and Setup
import logging  # Import the logging module for logging messages
import joblib  # Import joblib for saving and loading scikit-learn models
import numpy as np  # Import NumPy for numerical operations
import pandas as pd  # Import pandas for data manipulation
import tensorflow as tf  # Import TensorFlow for deep learning
from tensorflow import keras  # Import Keras from TensorFlow for building neural networks
import torch  # Import PyTorch if needed for deep learning with PyTorch
import sklearn.base  # Import scikit-learn base classes
from sklearn.linear_model import LinearRegression  # Import LinearRegression from scikit-learn
from sklearn.ensemble import RandomForestRegressor  # Import RandomForestRegressor from scikit-learn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  # Import GridSearchCV and RandomizedSearchCV for hyperparameter tuning
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # Import Keras callbacks for training
from tensorflow.keras.models import save_model as save_keras_model  # Import Keras save_model function
from tensorflow.keras.models import load_model as load_keras_model  # Import Keras load_model function
from kerastuner.engine.hypermodel import HyperModel  # Import HyperModel from Keras Tuner for hyperparameter tuning
from kerastuner.tuners import BayesianOptimization  # Import BayesianOptimization from Keras Tuner for hyperparameter tuning
from statsmodels.tsa.arima.model import ARIMA  # Import ARIMA from statsmodels for time series modeling
import statsmodels  # Import statsmodels if needed for statistical analysis
import xgboost as xgb  # Import XGBoost for gradient boosting
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, InputLayer, Flatten, BatchNormalization, Reshape
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Reshape, LSTM, Dense, Dropout, BatchNormalization, Flatten
import optuna
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Flatten, BatchNormalization, InputLayer
from tensorflow.keras.optimizers import Adam, RMSprop

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




# Section 2: Model Training Functions - Part 1

def train_model(X_train, y_train, model_type='linear_regression', hyperparameter_tuning=False, param_grid=None, n_iter=None):
    """
    Train a machine learning model based on the specified type.

    Args:
        X_train (DataFrame): Training features.
        y_train (Series): Training target.
        model_type (str): Type of model to train.
        hyperparameter_tuning (bool): Flag indicating whether to perform hyperparameter tuning.
        param_grid (dict or None): Hyperparameter grid for GridSearchCV or RandomizedSearchCV.
        n_iter (int or None): Number of iterations for RandomizedSearchCV.

    Returns:
        model: Trained model object.
    """
    logger.info(f"Training {model_type} model...")
    
    if hyperparameter_tuning:
        if model_type == 'linear_regression':
            model = GridSearchCV(LinearRegression(), param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
        elif model_type == 'random_forest':
            model = RandomizedSearchCV(RandomForestRegressor(), param_distributions=param_grid, n_iter=n_iter, scoring='neg_mean_squared_error', cv=5)
        else:
            logger.error(f"Unsupported model type for hyperparameter tuning: {model_type}")
            raise ValueError(f"Unsupported model type for hyperparameter tuning: {model_type}")
    else:
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
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")

    if hyperparameter_tuning:
        model.fit(X_train, y_train)
        best_model = model.best_estimator_
    else:
        model.fit(X_train, y_train)
        best_model = model

    logger.info(f"{model_type} model trained successfully.")
    return best_model


# Function for creating a customizable neural network model


def create_neural_network(X_train, y_train, sequence_length, features_shape):
    try:
        def objective(trial):
            # Hyperparameter definitions
            num_layers = trial.suggest_int('num_layers', 1, 3)
            dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
            units_per_layer = trial.suggest_categorical('units', [16, 32, 64, 128])
            lstm_units = trial.suggest_categorical('lstm_units', [16, 32, 64])

            # Model creation with the trial's current hyperparameters
            model = Sequential()

            # Adding LSTM layers
            for i in range(num_layers):
                if i == 0:
                    # First layer needs to specify input shape
                    model.add(LSTM(lstm_units, return_sequences=(num_layers > 1), input_shape=(sequence_length, features_shape)))
                else:
                    model.add(LSTM(lstm_units, return_sequences=(i < num_layers - 1)))
                model.add(Dropout(dropout_rate))

            # Adding Dense layers
            for _ in range(num_layers):
                model.add(Dense(units_per_layer, activation='relu'))
                model.add(Dropout(dropout_rate))
            
            # Output layer for regression
            model.add(Dense(1, activation='linear')) 

            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

            # Training the model
            history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32, verbose=0)

            # Objective: Minimize the validation loss
            validation_loss = np.min(history.history['val_loss'])
            return validation_loss

        # Create an Optuna study and optimize the objective
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)  # Adjust n_trials as needed

        # Extract the best parameters
        best_params = study.best_params

        # Rebuild the model with the best parameters
        final_model = Sequential()

        # Adding LSTM layers
        for i in range(best_params['num_layers']):
            if i == 0:
                final_model.add(LSTM(best_params['lstm_units'], return_sequences=(best_params['num_layers'] > 1), input_shape=(sequence_length, features_shape)))
            else:
                final_model.add(LSTM(best_params['lstm_units'], return_sequences=(i < best_params['num_layers'] - 1)))
            final_model.add(Dropout(best_params['dropout_rate']))

        # Adding Dense layers
        for _ in range(best_params['num_layers']):
            final_model.add(Dense(best_params['units'], activation='relu'))
            final_model.add(Dropout(best_params['dropout_rate']))
        
        # Output layer for regression
        final_model.add(Dense(1, activation='linear')) 

        final_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

        # Final training with the entire dataset
        final_history = final_model.fit(X_train, y_train, validation_split=0.2, 
                                        epochs=study.best_params.get('epochs', 10), 
                                        batch_size=study.best_params.get('batch_size', 32), 
                                        verbose=1)

        return final_model

    except Exception as e:
        error_message = f"Error training neural network: {str(e)}"
        print(error_message)
        return None


def optimize_model():
    print("Optimizing model with Optuna.")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)  # Adjust n_trials as needed

    best_params = study.best_params
    print(f"Best parameters found: {best_params}")

    use_lstm = 'use_lstm' in best_params and best_params['use_lstm']
    lstm_layers = [best_params['lstm_units']] if use_lstm else None
    dropout_rates = [best_params['dropout_rate']] * (best_params['num_layers'] + (1 if use_lstm else 0))

    final_model = create_neural_network(input_shape=(sequence_length, features.shape[1]), 
                                        timesteps=sequence_length if use_lstm else None,
                                        layers=[best_params['units']] * best_params['num_layers'], 
                                        lstm_layers=lstm_layers,
                                        dropout_rates=dropout_rates,
                                        optimizer=best_params['optimizer'])

    print("Final model training with optimized parameters.")
    final_history = final_model.fit(X_train, y_train, validation_split=0.2, 
                                    epochs=best_params['epochs'], 
                                    batch_size=best_params['batch_size'], 
                                    verbose=1)

    # Assuming you have a function to evaluate your model's performance:
    evaluate_model(final_model, X_test, y_test)
    print("Model optimization and training complete.")


# Function for creating an LSTM model

def create_lstm_model(input_shape, lstm_layers, dropout_rates, timesteps=1):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    # If your data is not already in the shape (batch_size, timesteps, features),
    # you need to reshape it. Here, we're assuming each feature should be treated
    # as a separate timestep. This is a simplistic approach and may need adjustment.
    if len(input_shape) == 1:
        # Reshape the input to have a "time" dimension of 1
        model.add(tf.keras.layers.Reshape((timesteps, input_shape[0] // timesteps)))
    
    for i, (lstm_units, dropout_rate) in enumerate(zip(lstm_layers, dropout_rates)):
        model.add(tf.keras.layers.LSTM(units=lstm_units, return_sequences=(i < len(lstm_layers) - 1)))
        if dropout_rate > 0:
            model.add(tf.keras.layers.Dropout(dropout_rate))
    
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        # Validate that required columns exist
        if 'date' not in data.columns or 'close' not in data.columns:
            raise ValueError("Missing required columns 'date' or 'close'.")
        return data
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        raise
    except ValueError as e:
        print(f"Data validation error: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

def train_arima_model(filepath, order=(5,1,0)):
    print(f"Attempting to load data from: {filepath}, Type: {type(filepath)}")
    try:
        print(f"Received filepath: {filepath}, Type: {type(filepath)}")  # Corrected syntax here
        data = load_data(filepath)  # Load and validate data
        train_size = int(len(data['close']) * 0.8)
        train, test = data['close'][0:train_size], data['close'][train_size:]
        
        history = list(train)
        predictions = []
        
        for t in range(len(test)):
            model = ARIMA(history, order=order)
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test.iloc[t]
            history.append(obs)
        
        # Evaluate forecasts
        error = mean_squared_error(test, predictions)
        print(f'Test MSE: {error}')
        
        # Plot forecasts against actual outcomes
        plt.figure(figsize=(12, 6))
        plt.plot(test.values, label='Actual')
        plt.plot(predictions, color='red', label='Predicted')
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Error in training ARIMA model: {e}")

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
    logger.info("Evaluating the model...")
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    evaluation_metrics = {'mean_squared_error': mse, 'r2_score': r2}
    logger.info(f"Model evaluation complete. Metrics: {evaluation_metrics}")
    return evaluation_metrics

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
    logger.info(f"Saving the model to {filename}...")
    if isinstance(model, sklearn.base.BaseEstimator):
        joblib.dump(model, filename + '.joblib')
    elif isinstance(model, keras.Model):
        save_keras_model(model, filename + '.h5')
    elif isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), filename + '.pth')
    else:
        logger.error("Model type not supported")
        raise ValueError("Model type not supported")
    logger.info(f"Model saved to {filename}")


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

        # Define hyperparameter grid for neural network
        param_grid_neural_network = {
            'units': [64, 128, 256],
            'num_layers': [1, 2, 3],
            'dropout': [True, False],
            'dropout_rate': [0.2, 0.3, 0.4],
            'learning_rate': [1e-4, 1e-3, 1e-2]
        }

        best_model = train_model(X_train, y_train, model_type='neural_network', hyperparameter_tuning=True,
                                 param_grid=param_grid_neural_network, n_iter=max_trials)

    elif model_type == 'random_forest':
        # Define hyperparameter grid for random forest
        param_grid_random_forest = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        best_model = train_model(X_train, y_train, model_type='random_forest', hyperparameter_tuning=True,
                                 param_grid=param_grid_random_forest, n_iter=max_trials)

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
    logger.info(f"Loading model from {file_path}...")
    if file_path.endswith('.joblib'):
        # Load a scikit-learn model
        loaded_model = joblib.load(file_path)
    elif file_path.endswith('.h5'):
        # Load a Keras model
        loaded_model = tf.keras.models.load_model(file_path)
    elif file_path.endswith('.pth'):
        # Load a PyTorch model
        # Assuming you have a function to create and define the PyTorch model
        loaded_model = create_and_define_pytorch_model()
        loaded_model.load_state_dict(torch.load(file_path))
        loaded_model.eval()
    else:
        logger.error("Unsupported model file format or extension")
        raise ValueError("Unsupported model file format or extension")
    logger.info("Model loaded successfully.")
    return loaded_model



def train_hist_gradient_boosting(data_file_path, scaler_type, target_column='close'):
    """
    Train a historical Gradient Boosting model.

    Args:
        data_file_path (str): Path to the CSV file containing the historical data.
        scaler_type (str): Type of scaler to use for data preprocessing ('standard' or other supported scalers).
        target_column (str, optional): The name of the target column in the dataset (default is 'close').

    Returns:
        sklearn.ensemble.GradientBoostingRegressor: The trained Gradient Boosting regressor model.
    """
    try:
        logger.info("Loading historical data...")
        # Load the data from the provided file path
        df = pd.read_csv(data_file_path)

        # Handle missing values by forward fill (you can choose other methods as needed)
        df.fillna(method='ffill', inplace=True)

        logger.info("Extracting features and target...")
        # Extract features and target
        X = df.drop(target_column, axis=1)  # Assuming the target column is labeled 'close'
        y = df[target_column]

        logger.info("Performing data scaling...")
        # Perform data scaling if needed (e.g., using StandardScaler or other scalers)
        if scaler_type == "standard":
            scaler = StandardScaler()
        else:
            logger.error(f"Unsupported scaler type: {scaler_type}")
            raise ValueError(f"Unsupported scaler type: {scaler_type}")

        X_scaled = scaler.fit_transform(X)

        logger.info("Splitting the dataset into training and testing sets...")
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        logger.info("Creating an instance of the GradientBoostingRegressor...")
        # Create an instance of the GradientBoostingRegressor
        gbr = GradientBoostingRegressor()

        logger.info("Training the model...")
        # Train the model
        gbr.fit(X_train, y_train)

        logger.info("Evaluating the model...")
        # Evaluate the model (you can add evaluation metrics here)
        train_score = gbr.score(X_train, y_train)
        test_score = gbr.score(X_test, y_test)

        logger.info(f"Training Score: {train_score}")
        logger.info(f"Testing Score: {test_score}")

        logger.info("Training complete.")
        # Return the trained model
        return gbr

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise  # Re-raise the exception to handle it elsewhere

# Example usage
# model = train_hist_gradient_boosting('path_to_your_data.csv', 'standard')

# Section ???: Final Remarks and Cleanup

if __name__ == "__main__":
    # Example usage of your script

    # Configure logging
    logging.basicConfig(filename='model_training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Specify the file path to your historical data CSV file
    data_file_path = 'path_to_your_data.csv'

    # Specify the type of scaler to use ('standard' or other supported scalers)
    scaler_type = 'standard'

    try:
        # Train a historical Gradient Boosting model
        trained_model = train_hist_gradient_boosting(data_file_path, scaler_type)

        # Evaluate the trained model (provide test data and labels)
        # Replace 'X_test' and 'y_test' with your actual test data and labels
        evaluation_metrics = evaluate_model(trained_model, X_test, y_test)

        # Save the trained model to a file
        model_filename = 'trained_model'
        save_model(trained_model, model_filename)

        # Load the saved model from the file
        loaded_model = load_model(model_filename)

        # Create an ensemble of models (provide a list of trained models)
        models_to_ensemble = [trained_model, loaded_model]  # Example list of models
        ensemble_model = create_ensemble_model(models_to_ensemble)

        # Use the ensemble model to make predictions (provide input data 'X')
        # Replace 'X' with your actual input data
        ensemble_predictions = ensemble_model.predict(X)

        # Print or log the ensemble predictions
        print("Ensemble Predictions:", ensemble_predictions)

        # Log success message
        logging.info("Script execution completed successfully.")

    except Exception as e:
        # Log any exceptions that occur during script execution
        logging.error(f"An error occurred: {str(e)}")

    finally:
        # Perform any necessary cleanup here
        pass

