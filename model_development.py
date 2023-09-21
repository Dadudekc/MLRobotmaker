import configparser

config = configparser.ConfigParser()
config_file_path = 'config.ini'




import logging



try:
    if config.read(config_file_path) == []:
        logging.error(f"Error reading the configuration file: {config_file_path}")
        # return None
    # return config
except Exception as e:
    logging.error(f"An exception occurred while reading the config file: {e}")
    # return None



# Load configuration from config.ini
config_file_path = 'config.ini'
config.read(config_file_path)
input_size = config.getint('Model', 'input_size', fallback=11)
hidden_layers = config.getint('Model', 'hidden_layers', fallback=4)
output_size = config.getint('Model', 'output_size', fallback=1)
batch_size = config.getint('Training', 'batch_size', fallback=50)
epochs = config.getint('Training', 'epochs', fallback=300)



import re
import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf


base_path = os.path.dirname(os.path.abspath(__file__))

import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,accuracy_score
from tensorflow.keras.models import load_model
from keras_tuner import HyperModel, RandomSearch


def read_config(config_file_path='config.ini'):
    config = ConfigParser()
    try:
        if config.read(config_file_path) == []:
            logging.error(f"Error reading the configuration file: {config_file_path}")
            return None
        return config
    except Exception as e:
        logging.error(f"An exception occurred while reading the config file: {e}")
        return None

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess_data(data):
    logging.info("Preprocessing data... ⚙️")
    logging.info("Missing values before filling:")
    logging.info(data.isna().sum())
    data['date'] = pd.to_datetime(data['date'])
    return data

def split_data(X, y, scaler_type='standard'):
    logging.info(f"Splitting data with {scaler_type} scaling... 🔄")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train.fillna(X_train.median(), inplace=True)
    X_test.fillna(X_test.median(), inplace=True)

    if scaler_type == 'standard':
        X_scaler = StandardScaler()
    elif scaler_type == 'power':
        X_scaler = PowerTransformer()

    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

    return X_train, X_test, y_train_scaled, y_test_scaled, X_scaler, y_scaler

class RegressionHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()
        model.add(Dense(units=hp.Int('units', 8, 64, 4, default=8),
                        activation='relu',
                        input_shape=[self.input_shape]))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=hp.Choice('optimizer', ['adam', 'sgd']),
                      loss='mse')
        return model


def save_model_and_predictions(model, model_filename, predictions_filename, y_test, y_pred, hyperparameters, predictions_df, mse, accuracy):
    # Construct the filename with hyperparameters and evaluation metrics
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_filename}_MSE_{mse:.4f}_ACC_{accuracy:.4f}_{current_time}.h5"
    model.save(model_filename)
    predictions_df.to_csv(predictions_filename, index=False)

def train_model(model, X_train, y_train, X_test, y_test, batch_size, epochs, model_filename, predictions_filename, scaler):
    logging.info("Training model... 🏋️‍♀️")

    checkpoint_path = "best_model_open.h5"
    checkpoint = ModelCheckpoint(checkpoint_path,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min')

    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[checkpoint])

    best_model = load_model(checkpoint_path)

    y_pred_scaled = best_model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test = y_test.reshape(-1)
    y_pred = y_pred.flatten()

    # Calculate MSE and accuracy
    mse = mean_squared_error(y_test, y_pred)
    y_pred_rounded = np.round(y_pred)  # Assuming y_test and y_pred are both continuous and you're rounding to the nearest integer
    accuracy = accuracy_score(y_test.astype(int), y_pred_rounded.astype(int))

    # Create a DataFrame for predictions
    predictions_df = pd.DataFrame({
        'y_test': y_test,
        'y_pred': y_pred
    })

    hyperparameters = {'learning_rate': 0.001, 'batch_size': 32}  # Adjust as needed
    save_model_and_predictions(best_model, model_filename, predictions_filename, y_test, y_pred, hyperparameters, predictions_df, mse, accuracy)


def evaluate_model(model, X_test, y_test_scaled, y_scaler):
    logging.info("Evaluating model... 📊")
    y_pred_scaled = model.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_test = y_test_scaled.reshape(-1)
    y_pred = y_pred.flatten()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"Mean Squared Error: {mse} 📉")
    logging.info(f"Mean Absolute Error: {mae} 📈")
    logging.info(f"R2 Score: {r2} 🎯")
    return mse

def main():
    # Check if predictions directory exists, if not, create it
    if not os.path.exists(predictions_directory):
        os.makedirs(predictions_directory)

    # Read config
    config = configparser.ConfigParser()
config.read(config_file_path)

# Define batch_size and epochs
batch_size = config.getint('Training', 'batch_size')
epochs = config.getint('Training', 'epochs')

# Define directories
models_directory = config.get('Paths', 'models_directory')
predictions_directory = "predictions"
data_directory = config.get('Paths', 'data_directory')

# Create directories for predictions and models
create_directory(predictions_directory)
create_directory(models_directory)

# Rest of your code, including the loop for processing data files
best_mse = float('inf')
best_model = None

for file in os.listdir(data_directory):
    try:
        if file.endswith(".csv"):
            logging.info(f"Processing {file}... 🔍")
            data = pd.read_csv(os.path.join(data_directory, file))
                
            for scaler_type in ['standard', 'power']:
                model_name = re.sub(r'_data_processed_data\.csv$', '', file) + '_' + scaler_type

                # Preprocess data
                processed_data = preprocess_data(data)
                X = processed_data[['open', 'high', 'low', 'volume', 'SMA_10', 'EMA_10', 'Price_RoC', 'Bollinger_High', 'Bollinger_Low', 'Bollinger_Mid']]
                y = processed_data['close']

                # Verify the shape of your input data
                print(f"Shape of X: {X.shape}")
                print(f"Shape of y: {y.shape}")

                # Split and scale data
                X_train, X_test, y_train_scaled, y_test_scaled, X_scaler, y_scaler = split_data(X, y, scaler_type)
                y_train_scaled = y_scaler.fit_transform(y_train_scaled.reshape(-1, 1))
                y_test_scaled = y_scaler.transform(y_test_scaled.reshape(-1, 1))

                # Build and train model
                hidden_layers = config.getint('Model', 'hidden_layers')
                output_size = config.getint('Model', 'output_size')
                hypermodel = RegressionHyperModel(input_shape=X_train.shape[1])
                tuner = RandomSearch(
                    hypermodel,
                    objective='val_loss',
                    max_trials=10,
                    executions_per_trial=2
                )
                tuner.search(
                    X_train,
                    y_train_scaled,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                    validation_data=(X_test, y_test_scaled)
                )
                best_model = tuner.get_best_models(num_models=1)[0]
                train_model(
                    best_model,
                    X_train,
                    y_train_scaled,
                    X_test,
                    y_test_scaled,
                    batch_size=batch_size,
                    epochs=epochs,
                    model_filename=os.path.join(models_directory, f"{model_name}_model.h5"),
                    predictions_filename=os.path.join(predictions_directory, f"{model_name}_predictions.csv"),
                    scaler=y_scaler
                )

                # Evaluate the model
                mse = evaluate_model(best_model, X_test, y_test_scaled, y_scaler)

                # Check if this model is the best model so far
                if mse < best_mse:
                    best_model = best_model
                    best_mse = mse

    except Exception as e:
        logging.error(f"Could not process {file}. Error: {e} ❌")

if __name__ == "__main__":
    main()
def build_model(input_shape):
    # Placeholder for building your model
    pass
