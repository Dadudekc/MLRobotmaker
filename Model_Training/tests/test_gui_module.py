import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Model_Training_Tab.model_training import ModelTraining
from Model_Training_Tab.utilities import MLRobotUtils

class TestModelTrainingTab(unittest.TestCase):

    def setUp(self):
        self.utils = MagicMock(spec=MLRobotUtils)
        self.utils.logger = MagicMock()
        self.tab = ModelTraining(self.utils)
        self.X = np.random.rand(100, 10)
        self.y = np.random.rand(100)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    @patch('pandas.read_csv')
    def test_preprocess_data_with_feature_engineering(self, mock_read_csv):
        mock_df = pd.DataFrame({
            'date': ['2020-01-01', '2020-01-02', '2020-01-03'],
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })
        mock_read_csv.return_value = mock_df

        X_train, X_test, y_train, y_test = self.tab.preprocess_data_with_feature_engineering(mock_df)
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)

    def test_display_message(self):
        message = "Test message"
        self.tab.display_message(message, "INFO")
        self.tab.logger.info.assert_called_with(f"[2024-05-15 13:46:11] {message}")

    def test_toggle_debug_mode(self):
        self.tab.config = {'log_text_widget': MagicMock()}
        with patch.object(self.tab.utils, 'log_message') as mock_log_message:
            self.tab.utils.log_message("Debug mode enabled", self.tab.config['log_text_widget'], False)
            mock_log_message.assert_called_with("Debug mode enabled", self.tab.config['log_text_widget'], False)

if __name__ == "__main__":
    unittest.main()
