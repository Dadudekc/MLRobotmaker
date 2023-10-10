#section 1: imports and initial setup
import traceback
import os
import configparser
import re
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from datetime import datetime
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from tensorflow.keras.models import load_model
from keras_tuner import HyperModel, RandomSearch

# Change working directory to the script's directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

config = configparser.ConfigParser()
config_file_path = 'config.ini'
try:
    if config.read(config_file_path) == []:
        logging.error(f"Error reading the configuration file: {config_file_path}")
        exit(1)  # Exit the script with an error code
except Exception as e:
    logging.error(f"An exception occurred while reading the config file: {e}")
    exit(1)

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#section 2: Utility functions

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_model_metadata(model_path, features):
    metadata = {
        'features': features
    }
    metadata_path = os.path.splitext(model_path)[0] + '.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

def load_model_metadata(model_path):
    metadata_path = os.path.splitext(model_path)[0] + '.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def preprocess_data(data):
    logging.info("Preprocessing data... ⚙️")
    logging.info("Missing values before filling:")
    logging.info(data.isna().sum())
    data['date'] = (pd.to_datetime(data['date']).view('int64') / 10**9).astype('int32')
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


#section 3: model training and evaluation functions

# Moved the TensorFlow function outside of any loops to optimize TensorFlow operations
@tf.function
def predict_with_model(model, input_data):
    return model(input_data)

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

def save_model_and_predictions_with_metadata(model, model_filename, predictions_filename, y_test, y_pred, scaler_type, hyperparameters, predictions_df, mse, accuracy, feature_names, correct_directions, y_test_diff, ticker, models_directory):
    # Construct the filename with hyperparameters and evaluation metrics
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate the directional accuracy
    directional_accuracy = correct_directions / len(y_test_diff) if len(y_test_diff) != 0 else 0
    
    # Abbreviation for scaler type
    scaler_abbrev = {
        'standard': 'std',
        'power': 'pwr'
    }.get(scaler_type, scaler_type)
    
    # New model naming convention
    model_name = f"{ticker}_A{directional_accuracy:.2f}_M{mse:.2f}_{scaler_abbrev}_{current_time}.h5"
    
    # Serialize the hyperparameters dictionary
    serialized_hyperparameters = json.dumps(hyperparameters)
    
    # Save the model
    model_path = os.path.join(models_directory, model_name)
    model.save(model_path)
    
    # Save the predictions
    predictions_df.to_csv(predictions_filename, index=False)
    
    # Extracting model architecture
    model_architecture = model.to_json()
    
    # Constructing metadata
    metadata = {
        "features": feature_names,
        "model_architecture": json.loads(model_architecture),
        "hyperparameters": serialized_hyperparameters
    }
    
    # Define metadata filename
    metadata_filename = model_path.replace(".h5", ".json")
    
    # Save metadata
    with open(metadata_filename, "w") as file:
        json.dump(metadata, file)

def train_model(model, X, y, batch_size, epochs, model_filename, predictions_filename, scaler, ticker, scaler_type, models_directory):
    logging.info("Training model... 🏋️‍♀️")
    
    # Split and scale data FIRST before referencing the split variables
    X_train, X_test, y_train_scaled, y_test_scaled, X_scaler, y_scaler = split_data(X, y, scaler_type)
    y_test = y_scaler.inverse_transform(y_test_scaled)

    best_model_path = f"{ticker}_best_model.h5"
    checkpoint_path = "model_checkpoints/" + ticker + "_checkpoint.h5"
    checkpoint = ModelCheckpoint(checkpoint_path,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min')

    # Ensure you're using y_train_scaled or y_train based on what you intend to use
    model.fit(X_train, y_train_scaled,
              validation_data=(X_test, y_test_scaled),
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[checkpoint])

    best_model = load_model(checkpoint_path)

    # Use the optimized TensorFlow function for prediction
    y_pred_scaled = predict_with_model(best_model, X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test = y_test.reshape(-1)
    y_pred = y_pred.flatten()

    # Calculate MSE and accuracy
    mse = mean_squared_error(y_test, y_pred)
    y_pred_rounded = np.round(y_pred)  # Assuming y_test and y_pred are both continuous and you're rounding to the nearest integer
    
    # Calculate differences for y_test and y_pred
    y_test_diff = np.diff(y_test)  # Calculate y_test_diff here
    y_pred_diff = np.diff(y_pred)  # Calculate y_pred_diff here
    
    # Check if y_pred_diff is not empty before proceeding
    if len(y_pred_diff) > 0:
        model_filename = f"{model_filename}_MSE_{'{:.4f}'.format(mse) if isinstance(mse, (int, float)) else mse}_ACC_{'{:.4f}'.format(directional_accuracy) if isinstance(directional_accuracy, (int, float)) else directional_accuracy}_{current_time}.h5"
        accuracy = accuracy_score(y_test.astype(int), y_pred_rounded.astype(int))

    # Create a DataFrame for predictions
    predictions_df = pd.DataFrame({
        'y_test': y_test,
        'y_pred': y_pred
    })

    hyperparameters = {'learning_rate': 0.001, 'batch_size': 32}  # Adjust as needed
    feature_names = X.columns.tolist()

    # Split and scale data
    X_train, X_test, y_train_scaled, y_test_scaled, X_scaler, y_scaler = split_data(X, y, scaler_type)
    correct_directions = np.sum(np.sign(y_test_diff) == np.sign(y_pred_diff))

    # Now pass y_test_diff and y_pred_diff to save_model_and_predictions_with_metadata
    save_model_and_predictions_with_metadata(
    best_model, model_filename, predictions_filename, y_test, y_pred, scaler_type, hyperparameters, predictions_df, mse, accuracy, feature_names, ticker, models_directory, correct_directions, y_test_diff, y_pred_diff
)

    print("Debugging train_model before save_model_and_predictions_with_metadata:")
    print("model_filename:", model_filename)
    print("predictions_filename:", predictions_filename)
    print("y_test:", y_test)
    print("y_pred:", y_pred)
    print("y_scaler:", y_scaler)
    print("ticker:", ticker)
    print("scaler_type:", scaler_type)
    print("models_directory:", models_directory)
    return mse, accuracy, y_pred, predictions_df


def directional_accuracy(y_true, y_pred, y_test_diff, y_pred_diff):
    """Accuracy of the prediction direction.
    This metric computes the percentage of times that y_pred and y_true have the same sign, indicating that the model
    predicted the correct movement direction.
    """
    correct_directions = np.sum(np.sign(y_test_diff) == np.sign(y_pred_diff))
    return correct_directions / len(y_test_diff) if len(y_test_diff) != 0 else 0

def evaluate_model(model, X_test, y_test_scaled, y_scaler):
    # Use the optimized TensorFlow function for prediction
    y_pred_scaled = predict_with_model(model, X_test)
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


# section 4: 

def main():
    # Define batch_size and epochs
    batch_size = config.getint('Training', 'batch_size')
    epochs = config.getint('Training', 'epochs')

    # Define directories
    model_directories = {
        'power': config.get('DIRECTORIES', 'PowerModelDirectory'),
        'standard': config.get('DIRECTORIES', 'StandardModelDirectory')
    }
    predictions_directory = "predictions"
    data_directory = config.get('Paths', 'data_directory')

    # Check if predictions directory exists, if not, create it
    if not os.path.exists(predictions_directory):
        os.makedirs(predictions_directory)
    
    # Create directories for predictions and models
    create_directory(predictions_directory)
    
    for dir_path in model_directories.values():   # This line should be indented to be inside main()
        create_directory(dir_path)


    # Rest of your code, including the loop for processing data files
    best_mse = float('inf')
    best_model = None  # Declare best_model here

    y_pred_diff = np.array([])  

    for file in os.listdir(data_directory):
        try:
            if file.endswith(".csv"):
                logging.info(f"Processing {file}... 🔍")
                data = pd.read_csv(os.path.join(data_directory, file))
                print("Extracted ticker from file:", file)
                ticker = os.path.basename(file).split("_")[0]

                for scaler_type in ['standard', 'power']:
                    models_directory = model_directories[scaler_type]

                    
                    model_name = re.sub(r'_data_processed_data\.csv$', '', file) + '_' + scaler_type
                    processed_data = preprocess_data(data)


                    # Read the CSV file as a DataFrame
                    data = pd.read_csv(os.path.join(data_directory, file))

                    # Verify the shape of your input data
                    print(f"Shape of data: {data.shape}")

                    # Assuming 'data' is your DataFrame after reading the CSV
                    all_columns = data.columns.tolist()

                    # Get the columns you want to use for X
                    columns_to_use = [col for col in all_columns]

                    # Now use those columns for processing
                    X = processed_data[columns_to_use]
                    y = processed_data['close']

                    # Verify the shape of your input data
                    print(f"Shape of X: {X.shape}")
                    print(f"Shape of y: {y.shape}")

                    # Split and scale data
                    X_train, X_test, y_train_scaled, y_test_scaled, X_scaler, y_scaler = split_data(X, y, scaler_type)
                    y_train_unscaled = y_scaler.inverse_transform(y_train_scaled)
                    y_test_unscaled = y_scaler.inverse_transform(y_test_scaled)
                    
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

                    # Check if any models were found by tuner
                    best_models = tuner.get_best_models(num_models=1)

                    if best_models:
                        best_model = best_models[0]
                        
                        # Train and evaluate the best model
                        print("Debugging main before train_model:")
                        print("model_filename:", os.path.join(models_directory, f"{model_name}_model.h5"))
                        print("predictions_filename:", os.path.join(predictions_directory, f"{model_name}_predictions.csv"))
                        print("X shape:", X.shape)
                        print("y shape:", y.shape)
                        print("ticker:", ticker)
                        print("scaler_type:", scaler_type)
                        print("models_directory:", models_directory)

                        # Initialize y_test_diff with an empty array or appropriate initial value
                        y_test_diff = None

                        mse, accuracy, y_pred, predictions_df = train_model(
                            best_model,
                            X,
                            y,
                            batch_size=batch_size,
                            epochs=epochs,
                            model_filename=os.path.join(models_directory, f"{model_name}_model.h5"),
                            predictions_filename=os.path.join(predictions_directory, f"{model_name}_predictions.csv"),
                            scaler=y_scaler,
                            scaler_type=scaler_type,
                            models_directory=models_directory,
                            ticker=ticker,  # Pass the ticker here
                            y_test_diff=y_test_diff,
                            y_pred_diff=y_pred_diff  # Pass the differences here
                        )

                        # Calculate differences for y_test and y_pred
                        y_test_diff = np.diff(y_test_unscaled)  # Calculate y_test_diff here
                        y_pred_diff = np.diff(y_pred_unscaled)  # Calculate y_pred_diff here

                        # Extracting hyperparameters from the tuner
                        hyperparameters = tuner.get_best_hyperparameters()[0].values

                        # Call save_model_and_predictions_with_metadata here
                        print("Debugging main after train_model:")
                        print("model_filename:", os.path.join(models_directory, f"{model_name}_model.h5"))
                        print("predictions_filename:", os.path.join(predictions_directory, f"{model_name}_predictions.csv"))
                        print("y_test_unscaled:", y_test_unscaled)
                        print("y_pred:", y_pred)
                        print("hyperparameters:", hyperparameters)
                        print("predictions_df shape:", predictions_df.shape)
                        print("mse:", mse)
                        print("accuracy:", accuracy)
                        print("columns_to_use:", columns_to_use)
                        print("Calling save_model_and_predictions_with_metadata with arguments:")
                        print("ticker:", ticker)
                        print("models_directory:", models_directory)

                        save_model_and_predictions_with_metadata(
                            best_model,
                            os.path.join(models_directory, f"{model_name}_model.keras"),
                            os.path.join(predictions_directory, f"{model_name}_predictions.csv"),
                            y_test_unscaled,
                            y_pred,
                            scaler_type,
                            hyperparameters,
                            predictions_df,
                            mse,
                            accuracy,
                            columns_to_use,
                            ticker=ticker,  # <-- add ticker here
                            models_directory=models_directory,
                            y_test_diff=y_test_diff
                        )


                    else:
                        # Handle the case where no best models were found
                        logging.error("No best models found for this configuration. Skipping further processing.")
                        continue  # Move to the next file

        except Exception as e:
            logging.error(f"Could not process {file}. Error: {e} ❌")
            logging.error(traceback.format_exc()) 

if __name__ == "__main__":
    main()
