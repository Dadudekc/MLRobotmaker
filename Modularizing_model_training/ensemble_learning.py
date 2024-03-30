# ensemble_learning.py

from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

class EnsembleModel:
    def __init__(self):
        self.models = [
            ('lr', LinearRegression()),
            ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.1)),
            ('lgbm', LGBMRegressor(n_estimators=100, learning_rate=0.1))
        ]
        self.ensemble_model = VotingRegressor(estimators=self.models)

    def train(self, X_train, y_train):
        self.ensemble_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.ensemble_model.predict(X_test)
