# continuous_learning.py

import time
import pandas as pd
from stable_baselines3 import PPO
from data_preprocessing import load_and_preprocess_data
from train_drl_model import train_drl_model
from backtest import backtest_drl_model

def continuous_learning(file_path, retrain_interval=86400, total_timesteps=10000):
    while True:
        print("Training model...")
        model, scaler = train_drl_model(file_path, total_timesteps)
        
        print("Backtesting model...")
        total_reward, final_balance, mfe, mae = backtest_drl_model(file_path, "ppo_trading_model")
        
        print(f"Total Reward: {total_reward}, Final Balance: {final_balance}, MFE: {mfe}, MAE: {mae}")
        
        time.sleep(retrain_interval)

if __name__ == "__main__":
    continuous_learning('path_to_your_data.csv')
