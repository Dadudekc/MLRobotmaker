# train_drl_model.py

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
from data_preprocessing import load_and_preprocess_data
from trading_env import TradingEnv

def train_drl_model(file_path, total_timesteps=10000):
    data, data_scaled, scaler = load_and_preprocess_data(file_path)
    env = DummyVecEnv([lambda: TradingEnv(data_scaled)])
    
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save("ppo_trading_model")
    
    return model, scaler

if __name__ == "__main__":
    model, scaler = train_drl_model('path_to_your_data.csv')
