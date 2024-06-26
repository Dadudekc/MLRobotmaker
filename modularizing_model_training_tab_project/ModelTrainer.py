import numpy as np
import pandas as pd
import asyncio
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.regularizers import l1_l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

class ModelTrainer:
    def __init__(self, optimizer=None, logger=None):
        self.optimizer = optimizer or Adam(learning_rate=1e-4)
        self.logger = logger
        if self.logger is None:
            import logging
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)

    async def train_model_async(self, X_train, y_train, X_val, y_val, model_type='neural_network', epochs=100):
        """ Asynchronously train a model based on the type and provided data. """
        model = await self.initialize_and_configure_model(model_type, input_shape=X_train.shape[1:], epochs=epochs)
        X_train, X_val = self.reshape_data(X_train, X_val, model_type)
        model = self.compile_and_train(model, X_train, y_train, X_val, y_val, epochs)
        mse, rmse, r2 = self.evaluate_model(model, X_val, y_val)
        self.logger.info(f"Validation MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
        return model, (mse, rmse, r2)

    async def initialize_and_configure_model(self, model_type, input_shape, epochs):
        """ Initialize and configure a machine learning model based on the type. """
        if model_type == "neural_network":
            model = Sequential([
                Dense(128, activation='relu', input_shape=input_shape),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dense(1)
            ])
        elif model_type == "LSTM":
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(input_shape[1], input_shape[2])),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1)
            ])
        elif model_type == "random_forest":
            model = await self.configure_random_forest_async()
        else:
            raise ValueError("Unsupported model type")
        model.compile(optimizer=self.optimizer, loss='mean_squared_error')
        return model

    async def configure_random_forest_async(self):
        """ Asynchronously configure and tune a Random Forest model. """
        from sklearn.ensemble import RandomForestRegressor
        import optuna
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
            return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        return study.best_estimator_

    def reshape_data(self, X_train, X_val, model_type):
        """ Reshape data based on the model requirements. """
        if model_type == "LSTM":
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        return X_train, X_val

    def compile_and_train(self, model, X_train, y_train, X_val, y_val, epochs):
        """ Compile and train the model using the specified optimizer and early stopping. """
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=32, callbacks=[early_stopping])
        return model

    def evaluate_model(self, model, X_val, y_val):
        """ Evaluate the model and return performance metrics. """
        y_pred_val = model.predict(X_val).flatten()
        mse = mean_squared_error(y_val, y_pred_val)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred_val)
        return mse, rmse, r2

# Usage example with asynchronous capabilities
trainer = ModelTrainer()
# Assuming X_train, y_train, X_val, y_val are defined and async is required
# loop = asyncio.get_event_loop()
# loop.run_until_complete(trainer.train_model_async(X_train, y_train, X_val, y_val))
