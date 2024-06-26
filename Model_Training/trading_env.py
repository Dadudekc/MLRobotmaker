# trading_env.py

import gym
from gym import spaces
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=1000):
        super(TradingEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0
        self.current_step = 0
        self.max_price = 0
        self.min_price = np.inf

        self.action_space = spaces.Discrete(3)  # [hold, buy, sell]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.data.shape[1],), dtype=np.float32)

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.max_price = 0
        self.min_price = np.inf
        return self._next_observation()

    def _next_observation(self):
        return self.data[self.current_step]

    def step(self, action):
        current_price = self.data[self.current_step, 0]  # Assuming 'Close' price is the first column
        self.max_price = max(self.max_price, current_price)
        self.min_price = min(self.min_price, current_price)

        reward = 0
        done = False

        if action == 1:  # Buy
            self.position += self.balance / current_price
            self.balance = 0
        elif action == 2:  # Sell
            self.balance += self.position * current_price
            self.position = 0
            reward = self.balance - self.initial_balance  # Calculate reward based on profit/loss

        self.current_step += 1

        if self.current_step >= len(self.data) - 1:
            done = True
            # Calculate MFE and MAE
            mfe = (self.max_price - current_price) / current_price
            mae = (current_price - self.min_price) / current_price
            reward += mfe - mae

        return self._next_observation(), reward, done, {}

    def render(self, mode='human'):
        profit = self.balance + self.position * self.data[self.current_step, 0] - self.initial_balance
        print(f'Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}, Profit: {profit}')
