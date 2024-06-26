import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

class DataPreprocessing:
    def __init__(self, utils):
        self.utils = utils

    def preprocess_data(self, data_file_path, scaler_type, model_type):
        data = pd.read_csv(data_file_path)
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data['days_since_reference'] = (data['date'] - data['date'].min()).dt.days
        if 'close' in data.columns:
            y = data['close']
            X = data.drop(columns=['close', 'date'])
        else:
            return None, None, None, None

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)

        scaler = self.get_scaler(scaler_type)
        X_scaled = scaler.fit_transform(X_imputed)
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        return X_train, X_val, y_train, y_val

    def get_scaler(self, scaler_type):
        scalers = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'Normalizer': Normalizer(),
            'MaxAbsScaler': MaxAbsScaler()
        }
        return scalers.get(scaler_type, StandardScaler())

    def load_and_preprocess_data(self, file_path):
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        
        # Calculate technical indicators (example: moving average)
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        data['RSI'] = self.calculate_rsi(data['Close'])
        
        # Fill missing values
        data.fillna(method='ffill', inplace=True)
        
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        
        return data, data_scaled, scaler

    def calculate_rsi(self, series, period=14):
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
