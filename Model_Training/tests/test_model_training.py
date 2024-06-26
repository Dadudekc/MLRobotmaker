import unittest
import tkinter as tk
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from sklearn.model_selection import train_test_split
from Model_Training_Tab.gui_module import ModelTrainingTab
from Model_Training_Tab.utilities import MLRobotUtils
from Model_Training_Tab.model_training import ModelTraining
from datetime import datetime
from keras.models import Sequential

class TestModelTrainingTab(unittest.TestCase):

    def setUp(self):
        # GUI-related setup
        self.root = tk.Tk()
        self.config = {'log_text_widget': MagicMock()}
        self.scaler_options = ['StandardScaler', 'MinMaxScaler', 'RobustScaler']
        self.tab = ModelTrainingTab(self.root, self.config, self.scaler_options)
        self.tab.utils = MLRobotUtils()
        self.tab.utils.log_message = MagicMock()
        self.tab.logger = MagicMock()
        self.tab.logger.info = MagicMock()
        self.tab.logger.error = MagicMock()

        # Model training-related setup
        self.logger = MagicMock()
        self.utils = MLRobotUtils()
        self.model_training = ModelTraining(self.logger)
        self.X = np.random.rand(100, 10)
        self.y = np.random.rand(100)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def test_initialization(self):
        self.assertIsInstance(self.tab.utils, MLRobotUtils)

    def test_browse_data_file(self):
        self.tab.data_file_entry.insert(0, "test_data.csv")
        self.assertEqual(self.tab.data_file_entry.get(), "test_data.csv")

    def test_start_training_without_data_file(self):
        self.tab.data_file_entry.delete(0, tk.END)
        with self.assertRaises(ValueError):  # Assuming ValueError is raised
            self.tab.start_training()

    def test_start_training_without_model_type(self):
        self.tab.model_type_var.set("")
        with self.assertRaises(ValueError):  # Assuming ValueError is raised
            self.tab.start_training()

    @patch.object(ModelTrainingTab, 'display_message')
    def test_toggle_debug_mode(self, mock_display_message):
        self.tab.toggle_debug_mode()
        mock_display_message.assert_called_with("Debug mode enabled", level="DEBUG")
        self.tab.toggle_debug_mode()
        mock_display_message.assert_called_with("Debug mode disabled", level="DEBUG")

    @patch('pandas.read_csv')
    def test_preprocess_data_with_feature_engineering(self, mock_read_csv):
        mock_df = pd.DataFrame({
            'date': ['2020-01-01', '2020-01-02', '2020-01-03'],
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })
        mock_read_csv.return_value = mock_df

        X_train, X_test, y_train, y_test = self.model_training.preprocess_data_with_feature_engineering(mock_df)
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)

    def test_display_message(self):
        message = "Test message"
        self.model_training.display_message(message, "INFO")
        self.logger.info.assert_called()
        args, _ = self.logger.info.call_args
        self.assertIn(message, args[0])

    @patch('tensorflow.keras.models.Sequential.fit')
    def test_train_neural_network(self, mock_fit):
        model_type = 'neural_network'
        epochs = 50
        trained_model = self.model_training.train_neural_network_or_lstm(
            self.X_train, self.y_train, self.X_val, self.y_val, model_type, epochs)
        trained_model.compile(optimizer=self.model_training.model_configs[model_type]['optimizer'], 
                              loss=self.model_training.model_configs[model_type]['loss'])
        mock_fit.assert_called()
        self.assertIsNotNone(trained_model)
        self.assertTrue(hasattr(trained_model, 'predict'))

    @patch('sklearn.ensemble.RandomForestRegressor.fit')
    def test_train_random_forest(self, mock_fit):
        model = RandomForestRegressor()
        model.fit(self.X_train, self.y_train)
        trained_model = self.model_training.train_random_forest(
            self.X_train, self.y_train, self.X_val, self.y_val)
        mock_fit.assert_called()
        self.assertIsNotNone(trained_model)
        self.assertTrue(hasattr(trained_model, 'predict'))

if __name__ == "__main__":
    unittest.main()
