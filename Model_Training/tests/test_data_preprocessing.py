import unittest
import pandas as pd
from unittest.mock import MagicMock, patch
from Model_Training_Tab.data_preprocessing import DataPreprocessing

class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        self.utils = MagicMock()
        self.data_preprocessing = DataPreprocessing(self.utils)

    @patch('pandas.read_csv')
    def test_preprocess_data(self, mock_read_csv):
        mock_df = pd.DataFrame({
            'date': ['2020-01-01', '2020-01-02'],
            'close': [100, 101],
            'volume': [1000, 1100]
        })
        mock_read_csv.return_value = mock_df

        X_train, X_test, y_train, y_test = self.data_preprocessing.preprocess_data('test_data.csv', 'StandardScaler', 'neural_network')
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)

if __name__ == "__main__":
    unittest.main()
