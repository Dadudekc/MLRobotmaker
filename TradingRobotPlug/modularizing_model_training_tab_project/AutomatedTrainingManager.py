import os
import threading
import schedule
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AutomatedTrainingManager:
    def __init__(self, config):
        self.config = self.validate_config(config)
        self.logger = logging.getLogger(__name__)

    def validate_config(self, config):
        """Ensure all required configuration parameters are provided."""
        required_keys = {"Data": ["file_path"], "Model": ["model_type", "epochs"], "schedule_dropdown": None}
        for key, subkeys in required_keys.items():
            if key not in config:
                raise ValueError(f"Configuration must include {key}.")
            if subkeys:
                for subkey in subkeys:
                    if subkey not in config[key]:
                        raise ValueError(f"Configuration for '{key}' must include {subkey}.")
        return config

    def display_message(self, message, level=logging.INFO):
        """Log messages at specified logging level."""
        self.logger.log(level, message)

    def start_automated_training(self):
        """Schedule automated training based on the configured interval."""
        interval = self.config.get("schedule_dropdown").lower()
        timing_map = {"daily": "day", "weekly": "week", "monthly": "month"}
        if interval not in timing_map:
            raise ValueError("Invalid scheduling interval provided.")

        schedule.every().__getattribute__(timing_map[interval]).at("10:00").do(self.run_automated_training_tasks)
        self.display_message(f"Automated training scheduled to run every {interval} at 10:00 AM.")
        threading.Thread(target=self.run_schedule, daemon=True).start()

    def run_automated_training_tasks(self):
        """Execute training tasks according to the schedule."""
        file_path = self.config["Data"]["file_path"]
        model_type = self.config["Model"]["model_type"]
        epochs = self.config["Model"]["epochs"]

        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"The data file path specified does not exist: {file_path}")

            self.display_message(f"Training {model_type} model for {epochs} epochs.")
            if self.train_model(file_path, model_type, epochs):
                self.display_message("Model training completed successfully.")
            else:
                self.display_message("Model training encountered issues.", level=logging.ERROR)
        except Exception as e:
            self.display_message(f"Failed during training: {e}", level=logging.ERROR)

    def train_model(self, file_path, model_type, epochs):
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data(file_path)
        model = self.build_model(X_train.shape[1])

        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
            batch_size=32,
            verbose=1
        )
        loss = model.evaluate(X_test, y_test, verbose=0)
        self.logger.info(f"Test loss: {loss}")
        return True

    def load_and_preprocess_data(self, file_path):
        data = pd.read_csv(file_path)
        X = data.drop('target', axis=1)
        y = data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

    def build_model(self, input_shape):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_shape,)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(1)  # Assuming a regression task
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def run_schedule(self):
        while True:
            schedule.run_pending()
            threading.sleep(1)

# Usage example
config = {
    "Data": {"file_path": "path/to/data.csv"},
    "Model": {"model_type": "neural_network", "epochs": 10},
    "schedule_dropdown": "daily"
}
training_manager = AutomatedTrainingManager(config)
training_manager.start_automated_training()
