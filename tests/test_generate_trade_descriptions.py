#test_generate_trade_descriptions.py

import pandas as pd
import random

def generate_trade_descriptions(num_trades=1000):
    stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'FB']
    actions = ['buy', 'sell']
    reasons = ['technical analysis', 'market sentiment', 'news release', 'earnings report', 'price action']
    outcomes = ['profit', 'loss']

    data = []
    for _ in range(num_trades):
        stock = random.choice(stocks)
        action = random.choice(actions)
        reason = random.choice(reasons)
        outcome = random.choice(outcomes)
        quantity = random.randint(1, 100)

        description = f"Decided to {action} {quantity} shares of {stock} due to {reason}."
        data.append([description, outcome])

    return pd.DataFrame(data, columns=['Description', 'Outcome'])

def main():
    num_trades = 1000  # Set the number of simulated trades
    trade_data = generate_trade_descriptions(num_trades)
    trade_data.to_csv('simulated_trade_data.csv', index=False)
    print(f"Generated {num_trades} simulated trade descriptions.")

if __name__ == "__main__":
    main()
