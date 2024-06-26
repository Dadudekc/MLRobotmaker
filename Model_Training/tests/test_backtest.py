import os
import sys
import unittest
import yfinance as yf
# Add the project directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtest import fetch_stock_data, backtest_drl_model, plot_backtest_results

class TestBacktest(unittest.TestCase):

    def setUp(self):
        self.ticker = "TSLA"
        self.start_date = "2021-01-01"
        self.end_date = "2022-01-01"
        self.model_path = "models/ppo_trading_model"
        self.data = fetch_stock_data(self.ticker, self.start_date, self.end_date)

    def test_fetch_stock_data(self):
        data = fetch_stock_data(self.ticker, self.start_date, self.end_date)
        self.assertFalse(data.empty)
        self.assertIn('Close', data.columns)

    def test_backtest_drl_model(self):
        total_reward, final_balance, mfe, mae = backtest_drl_model(self.data, self.model_path)
        self.assertIsInstance(total_reward, float)
        self.assertIsInstance(final_balance, float)
        self.assertIsInstance(mfe, float)
        self.assertIsInstance(mae, float)

    def test_plot_backtest_results(self):
        step_rewards = [1, 2, 3, 4, 5]
        step_prices = [100, 105, 110, 115, 120]
        plot_backtest_results(step_rewards, step_prices)
        # Ensure no exceptions raised during plotting

if __name__ == "__main__":
    unittest.main()
