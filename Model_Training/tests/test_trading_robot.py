import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
from data_preprocessing import load_and_preprocess_data, calculate_rsi
from trading_env import TradingEnv
from train_drl_model import train_drl_model
from backtest import backtest_drl_model
from continuous_learning import continuous_learning

class TestTradingRobot(unittest.TestCase):

    def setUp(self):
        self.data_file_path = 'test_data.csv'
        self.mock_data = pd.DataFrame({
            'Date': pd.date_range(start='1/1/2022', periods=100),
            'Close': np.random.rand(100) * 100,
            'Volume': np.random.randint(1000, 10000, size=100)
        })
        self.mock_data.to_csv(self.data_file_path, index=False)

    def test_load_and_preprocess_data(self):
        data, data_scaled, scaler = load_and_preprocess_data(self.data_file_path)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertTrue(len(data_scaled) > 0)
        self.assertIsNotNone(scaler)

    def test_calculate_rsi(self):
        rsi = calculate_rsi(self.mock_data['Close'])
        self.assertEqual(len(rsi), len(self.mock_data['Close']))

    def test_trading_env_initialization(self):
        data, data_scaled, scaler = load_and_preprocess_data(self.data_file_path)
        env = TradingEnv(data_scaled)
        self.assertEqual(env.initial_balance, 1000)
        self.assertEqual(env.current_step, 0)
        self.assertEqual(env.position, 0)

    def test_trading_env_step(self):
        data, data_scaled, scaler = load_and_preprocess_data(self.data_file_path)
        env = TradingEnv(data_scaled)
        obs = env.reset()
        self.assertEqual(len(obs), data_scaled.shape[1])
        obs, reward, done, _ = env.step(1)
        self.assertFalse(done)

    @patch('train_drl_model.PPO.learn')
    @patch('train_drl_model.PPO.save')
    def test_train_drl_model(self, mock_save, mock_learn):
        model, scaler = train_drl_model(self.data_file_path, total_timesteps=1000)
        mock_learn.assert_called_once()
        mock_save.assert_called_once()
        self.assertIsNotNone(model)
        self.assertIsNotNone(scaler)

    @patch('backtest.PPO.load')
    def test_backtest_drl_model(self, mock_load):
        mock_load.return_value.predict.return_value = (np.array([0]), None)
        total_reward, final_balance, mfe, mae = backtest_drl_model(self.data_file_path, 'ppo_trading_model')
        self.assertIsInstance(total_reward, float)
        self.assertIsInstance(final_balance, float)
        self.assertIsInstance(mfe, float)
        self.assertIsInstance(mae, float)

    @patch('continuous_learning.train_drl_model')
    @patch('continuous_learning.backtest_drl_model')
    @patch('time.sleep', return_value=None)
    def test_continuous_learning(self, mock_sleep, mock_backtest, mock_train):
        mock_train.return_value = (None, None)
        mock_backtest.return_value = (0, 0, 0, 0)
        with patch('builtins.print') as mocked_print:
            continuous_learning(self.data_file_path, retrain_interval=1, total_timesteps=1000)
            self.assertTrue(mock_train.called)
            self.assertTrue(mock_backtest.called)
            mocked_print.assert_any_call("Training model...")
            mocked_print.assert_any_call("Backtesting model...")

if __name__ == '__main__':
    unittest.main()
