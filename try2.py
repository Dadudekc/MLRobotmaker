import os
import logging
import configparser
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

class ConfigHandler:
    def __init__(self, config_file_path='config.ini'):
        self.config_file_path = config_file_path
        self.config = configparser.ConfigParser()
        self.read_config()
    
    def read_config(self):
        try:
            if self.config.read(self.config_file_path) == []:
                logging.error(f"Error reading the configuration file: {self.config_file_path}")
                exit(1)
        except Exception as e:
            logging.error(f"An exception occurred while reading the config file: {e}")
            exit(1)
        
    def get(self, section, key):
        return self.config.get(section, key)

class Utils:
    @staticmethod
    def create_directory(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
class DataHandler:
    @staticmethod
    def preprocess_data(data, config_handler):
        logging.info("Preprocessing data... ⚙️")
        logging.info("Missing values before filling:")
        logging.info(data.isnull().sum())
        data = data.fillna(method='ffill').fillna(method='bfill')
        logging.info("Missing values after filling:")
        logging.info(data.isnull().sum())
        
        scaler_type = config_handler.get("preprocessing", "scaler_type")
        if scaler_type == 'StandardScaler':
            scaler = StandardScaler()
        elif scaler_type == 'MinMaxScaler':
            scaler = MinMaxScaler()
        else:
            logging.error(f"Unknown scaler type specified: {scaler_type}")
            exit(1)
        
        data_scaled = scaler.fit_transform(data)
        data = pd.DataFrame(data_scaled, columns=data.columns)
        
        columns_to_use = ['column1', 'column2']  # Placeholder list
        data = data[columns_to_use]
        
        return data

    @staticmethod
    def split_data(data):
        X_train, X_test, y_train, y_test = train_test_split(
            data.drop('target_column', axis=1),
            data['target_column'],
            test_size=0.2,
            random_state=42
        )
        return X_train, X_test, y_train, y_test

class ModelHandler:
    @staticmethod
    def train_model(X_train, y_train, config_handler):
        n_layers = int(config_handler.get("model", "n_layers"))
        n_units = int(config_handler.get("model", "n_units"))
        
        model = Sequential()
        for _ in range(n_layers):
            model.add(Dense(n_units, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        
        epochs, batch_size, validation_split = 10, 32, 0.2
        callbacks = [EarlyStopping(monitor='val_loss', patience=5), 
                     ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                  validation_split=validation_split, callbacks=callbacks)
        
        return model

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, mae, r2

    @staticmethod
    def save_model_and_predictions_with_metadata(model, model_path, predictions_df, 
                                                 y_test_unscaled, y_pred, scaler_type,
                                                 hyperparameters, mse, accuracy, 
                                                 columns_to_use, ticker, models_directory):
        model.save(model_path)

def main():
    config_handler = ConfigHandler()
    
    data_directory_path = config_handler.get("data", "directory_path")
    all_data = load_all_data_files(data_directory_path)
    
    for data in all_data:
        processed_data = DataHandler.preprocess_data(data, config_handler)
        X_train, X_test, y_train, y_test = DataHandler.split_data(processed_data)
        model = ModelHandler.train_model(X_train, y_train, config_handler)
        mse, mae, r2 = ModelHandler.evaluate_model(model, X_test, y_test)
        ModelHandler.save_model_and_predictions_with_metadata(
            model, 
            "path_to_save_model", 
            "predictions_df", 
            "y_test_unscaled", 
            "y_pred", 
            "scaler_type",
            "hyperparameters", 
            mse, 
            mae, 
            "columns_to_use", 
            "ticker", 
            "models_directory"
        )

if __name__ == '__main__':
    main()