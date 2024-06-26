import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import yfinance as yf

def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def backtest_drl_model(data, model_path, transaction_cost=0.001):
    # Load the trading data
    data['Date'] = pd.to_datetime(data.index)
    data.set_index('Date', inplace=True)

    # Create the trading environment
    env = make_vec_env(lambda: TradingEnv(data), n_envs=1)

    # Load the trained model
    model = PPO.load(model_path)

    # Run the backtest
    obs = env.reset()
    done = False
    total_reward = 0
    final_balance = 0
    prices = []
    rewards = []
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        final_balance = info[0]['balance']
        prices.append(info[0]['price'])
        rewards.append(total_reward)

    # Calculate MFE and MAE
    prices = np.array(prices)
    mfe = np.max(prices) - prices[0]  # Maximum Favorable Excursion
    mae = prices[0] - np.min(prices)  # Maximum Adverse Excursion

    return total_reward, final_balance, mfe, mae

def plot_backtest_results(step_rewards, step_prices):
    plt.figure(figsize=(12, 6))
    
    # Plot the rewards
    plt.subplot(2, 1, 1)
    plt.plot(step_rewards, label='Cumulative Reward')
    plt.title('Backtest Results')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    
    # Plot the prices
    plt.subplot(2, 1, 2)
    plt.plot(step_prices, label='Stock Price', color='orange')
    plt.xlabel('Steps')
    plt.ylabel('Stock Price')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

class TradingEnv:
    # Define the trading environment
    def __init__(self, data):
        self.data = data
        self.current_step = 0
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_reward = 0
        self.done = False
        self.price = self.data['Close'].iloc[self.current_step]

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_reward = 0
        self.done = False
        self.price = self.data['Close'].iloc[self.current_step]
        return self._get_observation()

    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        self.price = self.data['Close'].iloc[self.current_step]
        reward = self._calculate_reward()
        self.total_reward += reward
        done = self.current_step >= len(self.data) - 1
        info = {'balance': self.balance, 'price': self.price}
        return self._get_observation(), reward, done, info

    def _take_action(self, action):
        action_type = action[0]
        amount = action[1]
        
        if action_type == 0:  # Buy
            total_possible = self.balance // self.price
            shares_bought = total_possible * amount
            cost = shares_bought * self.price * (1 + transaction_cost)
            self.balance -= cost
            self.shares_held += shares_bought
        elif action_type == 1:  # Sell
            shares_sold = self.shares_held * amount
            self.balance += shares_sold * self.price * (1 - transaction_cost)
            self.shares_held -= shares_sold

    def _calculate_reward(self):
        current_value = self.shares_held * self.price + self.balance
        reward = current_value - self.total_reward
        return reward

    def _get_observation(self):
        return np.array([self.balance, self.shares_held, self.price])
