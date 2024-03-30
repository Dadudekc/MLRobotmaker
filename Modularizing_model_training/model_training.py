import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
import asyncio
from concurrent.futures import ThreadPoolExecutor
from Utilities.shared_data_store import SharedDataStore, Observer
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from kafka import KafkaConsumer
import json
import optuna
from keras.callbacks import Callback
import plotly.figure_factory as ff
from config import Config
from logging_utils import setup_logging, log_message
from data_preprocessing import load_data, clean_data, generate_features, split_data
from ensemble_learning import EnsembleModel
from sklearn.metrics import mean_squared_error

def evaluate_ensemble(ensemble, X_test, y_test):
    predictions = ensemble.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Ensemble MSE: {mse}")

# Assuming EnsembleModel takes a list of (name, model) tuples
ensemble_model = EnsembleModel([(name, model) for name, model in trained_models.items()])

# Train the ensemble model (this step may vary depending on your EnsembleModel implementation)
ensemble_model.train(X_train, y_train)

# Evaluate the ensemble model
evaluate_ensemble(ensemble_model, X_test, y_test)


def train_model(model_name, X_train, y_train):
    if model_name == 'RandomForest':
        model = RandomForestRegressor(**Config.MODEL_PARAMS['RandomForest'])
    elif model_name == 'XGBoost':
        model = XGBRegressor(**Config.MODEL_PARAMS['xgboost'])
    elif model_name == 'LightGBM':
        model = LGBMRegressor(**Config.MODEL_PARAMS['lightgbm'])
    elif model_name == 'NeuralNetwork':
        model = Sequential()
        for layer in Config.MODEL_PARAMS['neural_network']['layers']:
            model.add(Dense(**layer))
        model.compile(optimizer=Config.MODEL_PARAMS['neural_network']['optimizer'], loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=Config.MODEL_PARAMS['neural_network']['epochs'], batch_size=Config.MODEL_PARAMS['neural_network']['batch_size'])
        return model
    else:
        raise ValueError("Model type not supported.")
    
    model.fit(X_train, y_train)
    return model

# Train individual models
models = ['RandomForest', 'XGBoost', 'LightGBM']
trained_models = {model_name: train_model(model_name, X_train, y_train) for model_name in models}

# Initialize logging
setup_logging(log_level=Config.LOGGING_CONFIG['log_level'], log_file_path=Config.LOGGING_CONFIG['file_path'])
log_message("Logging initialized in model_training module.", logging.INFO)

# Load and preprocess data
log_message("Loading and preprocessing data...", logging.INFO)
data = load_data(Config.DATA_PATH)
data = clean_data(data)
data = generate_features(data)
X_train, X_test, y_train, y_test = split_data(data, test_size=0.2, random_state=42)

# Continue with model training using preprocessed data...

def plot_confusion_matrix(y_true, y_pred):
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig = ff.create_annotated_heatmap(z=cm, colorscale='Viridis')
    fig.show()

class CustomMetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Implement your custom logic here
        if logs.get('loss') < 0.1:  # Example condition
            print("Custom stopping condition met")
            self.model.stop_training = True

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        # Add other parameters here
    }
    model = RandomForestRegressor(**params)
    return cross_val_score(model, X_train, y_train, n_jobs=-1, cv=3).mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

def consume_real_time_data(topic_name):
    consumer = KafkaConsumer(topic_name,
                             bootstrap_servers=['localhost:9092'],
                             value_deserializer=lambda x: json.loads(x.decode('utf-8')))
    for message in consumer:
        data = message.value
        # Process your data here

def feature_selection_lgbm(X_train, y_train):
    model = LGBMRegressor().fit(X_train, y_train)
    # Get feature importances
    importances = model.feature_importances_
    # Keep only the top N features
    N = 20  # adjust N based on your needs
    indices = np.argsort(importances)[-N:]
    return X_train.columns[indices]

# Use in preprocessing
selected_features = feature_selection_lgbm(X_train, y_train)
X_train = X_train[selected_features]
X_test = X_test[selected_features]

def initialize_nn_model(input_shape, params):
    model = Sequential()
    for layer in params.get('layers', []):
        if layer['type'] == 'Dense':
            model.add(Dense(layer['units'], activation=layer['activation'], input_shape=input_shape))
        # Add more layer types as needed
    model.compile(optimizer=params.get('optimizer', 'adam'), loss='mean_squared_error')
    return model



class ModelTrainingObserver(Observer):
    async def update(self, message, event_type):
        # Improved notification with dynamic handling based on event type
        if event_type == "model_update":
            print(f"Model Update Notification: {message}")
        elif event_type == "error":
            print(f"Error Notification: {message}")
        else:
            print(f"Notification: {message}, Type: {event_type}")

class ModelTrainer:
    def __init__(self, shared_data_store, dataset_name, gui_display):
        self.shared_data_store = shared_data_store
        self.dataset_name = dataset_name
        self.model = None
        self.model_history = []  # Store tuples of (model_type, mse, r2)
        self.loop = asyncio.get_event_loop()
        self.executor = ThreadPoolExecutor()
        self.gui_display = gui_display  # GUI display function or widget

    async def gui_message(self, message):
        if asyncio.iscoroutinefunction(self.gui_display):
            await self.gui_display(message)
        else:
            self.loop.call_soon_threadsafe(self.gui_display, message)

    async def preprocess_data(self, dataset):
        # Asynchronous data preprocessing before training
        # This method can be expanded with more preprocessing steps as needed
        dataset = dataset.dropna()  # Example preprocessing step
        return dataset

    async def train_model(self, model_type, params=None):
        # Fetch model parameters dynamically from SharedDataStore
        n_estimators = self.shared_data_store.get_configuration('n_estimators', 100)
        max_depth = self.shared_data_store.get_configuration('max_depth', -1)
        learning_rate = self.shared_data_store.get_configuration('learning_rate', 0.1)
        num_leaves = self.shared_data_store.get_configuration('num_leaves', 31)
        
        # Fetch the dataset
        dataset = await self.loop.run_in_executor(self.executor, self.shared_data_store.get_dataset, self.dataset_name)
        if dataset is None:
            await self.shared_data_store.notify_observers("Dataset not found.", "error")
            return

        # Run preprocessing in an executor
        dataset = await self.preprocess_data(dataset)

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(dataset.drop('target', axis=1), dataset['target'], test_size=0.2, random_state=42)

        # Initialize model based on type
        model = self.initialize_model(model_type, X_train, {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves
        })

        if model is None:
            await self.shared_data_store.notify_observers(f"Unsupported model type {model_type}.", "error")
            return

        # Training with error handling
        try:
            if model_type in ['neural_network', 'LSTM']:
                await self.loop.run_in_executor(None, lambda: model.fit(X_train, y_train, epochs=params.get('epochs', 10), batch_size=params.get('batch_size', 32), verbose=1))
            else:
                model.fit(X_train, y_train)
        except Exception as e:
            await self.shared_data_store.notify_observers(f"Error during training: {str(e)}", "error")
            return

        self.model = model
        await self.shared_data_store.notify_observers(f"Model {model_type} trained with dataset {self.dataset_name}", "model_update")


    def initialize_model(self, model_type, X_train, params):
        # LightGBM Model
        if model_type == 'lightgbm':
            model = lgb.LGBMRegressor(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                num_leaves=params.get('num_leaves', 31),
                max_depth=params.get('max_depth', -1)
            )
        # XGBoost Model
        elif model_type == 'xgboost':
            model = XGBRegressor(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 6),
                objective='reg:squarederror'
            )
        # Neural Network and LSTM Initialization remains the same
        elif model_type == 'neural_network':
            model = Sequential([
                Dense(params.get('units', 64), activation='relu', input_shape=(X_train.shape[1],)),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
        elif model_type == 'LSTM':
            model = Sequential([
                LSTM(params.get('units', 50), input_shape=(1, X_train.shape[1])),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
        else:
            return None
        return model


    async def evaluate_model(self, model_type):
        if self.model is None:
            await self.gui_message("No model has been trained yet. Let's train one!")
            return

        dataset = await self.loop.run_in_executor(self.executor, self.shared_data_store.get_dataset, self.dataset_name)
        X_test = dataset.drop('target', axis=1)
        y_test = dataset['target']
        X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1])) if model_type == 'LSTM' else X_test

        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        self.model_history.append((model_type, mse, r2))

        # Compare with last model if available
        if len(self.model_history) > 1:
            last_model_type, last_mse, last_r2 = self.model_history[-2]

            mse_improvement = last_mse - mse
            r2_improvement = r2 - last_r2

            if mse_improvement > 0 and r2_improvement > 0:
                await self.gui_message(f"Great news! Your new {model_type} model is an improvement over the last {last_model_type} model.")
                await self.gui_message(f"It's more accurate, which means it makes fewer mistakes when predicting.")
            elif mse_improvement <= 0 or r2_improvement <= 0:
                await self.gui_message("This new model didn't perform better than the last one, but that's okay! Learning which models don't work well is part of the process.")
        else:
            await self.gui_message("This is the first model trained. Let's use it as a benchmark for future models!")

        await self.gui_message(f"Evaluation results - MSE: {mse:.2f}, R2: {r2:.2f}. MSE tells us how close the predictions are to the actual values on average, while R2 shows the percentage of the target variable's variance explained by the model.")


# Example usage
shared_data_store = SharedDataStore()
shared_data_store.register_observer(ModelTrainingObserver(), interest="model_update")
trainer = ModelTrainer(shared_data_store, "sample_dataset_name")
# Note: Ensure that the event loop is running and can handle the async call appropriately.
asyncio.run(trainer.train_model("linear_regression"))
