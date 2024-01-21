import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, MaxAbsScaler

# Trade Class
class Trade:
    def __init__(self, symbol, size, order_type, price=None, stop_price=None):
        self.symbol = symbol
        self.size = size
        self.order_type = order_type
        self.price = price
        self.stop_price = stop_price

# Portfolio Class
class Portfolio:
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.positions = {}
        self.cash = initial_capital
        self.history = []
        self.total_value = initial_capital
        self.trade_history = []

    def execute_trade(self, trade, price):
        trade_cost = trade.size * price
        self.cash -= trade_cost
        self.positions[trade.symbol] = self.positions.get(trade.symbol, 0) + trade.size
        self.history.append({
            'symbol': trade.symbol, 
            'size': trade.size, 
            'price': price, 
            'portfolio_value': self.total_value
        })
        trade_outcome = {
            'symbol': trade.symbol,
            'size': trade.size,
            'entry_price': price,
            'exit_price': None,
            'profit_loss': 0,
            'outcome': 'open'
        }
        self.trade_history.append(trade_outcome)

    def update_portfolio_value(self, current_price):
        self.total_value = self.cash + sum(size * current_price.get(symbol, 0) for symbol, size in self.positions.items())

    def update_trade_outcome(self, symbol, exit_price):
        for trade in self.trade_history:
            if trade['symbol'] == symbol and trade['outcome'] == 'open':
                trade['exit_price'] = exit_price
                trade['profit_loss'] = (exit_price - trade['entry_price']) * trade['size']
                trade['outcome'] = 'win' if trade['profit_loss'] > 0 else 'loss'

    def calculate_trade_analytics(self):
        closed_trades = [trade for trade in self.trade_history if trade['outcome'] != 'open']
        wins = [trade for trade in closed_trades if trade['outcome'] == 'win']
        losses = [trade for trade in closed_trades if trade['outcome'] == 'loss']
        total_trades = len(closed_trades)
        win_ratio = len(wins) / total_trades if total_trades > 0 else 0
        avg_win = sum(trade['profit_loss'] for trade in wins) / len(wins) if wins else 0
        avg_loss = sum(trade['profit_loss'] for trade in losses) / len(losses) if losses else 0

        return {
            'total_trades': total_trades,
            'win_ratio': win_ratio,
            'average_win': avg_win,
            'average_loss': avg_loss
        }

# MLModelBacktest Class
class MLModelBacktest:
    def __init__(self, data_file_path, model_path, scaler_type, window_size=5, n_splits=3):
        self.model = joblib.load(model_path)
        self.portfolio = Portfolio(initial_capital=10000)
        self.historical_data = pd.read_csv(data_file_path)
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocess_data(
            data_file_path, scaler_type, window_size, n_splits)
        
    def preprocess_data(self, data_file_path, scaler_type, model_type, epochs, window_size=5, n_splits=3):
        try:
            # Load the dataset
            data = pd.read_csv(data_file_path)

            # Ensure the dataset has the required columns
            required_columns = {'open', 'high', 'low', 'close', 'volume'}
            if not required_columns.issubset(data.columns):
                raise ValueError("DataFrame must contain OHLCV columns.")

            # Handle missing values
            data = data.dropna()

            # Feature extraction from date column (if present)
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data['day_of_week'] = data['date'].dt.dayofweek
                data['month'] = data['date'].dt.month
                data['year'] = data['date'].dt.year

            # Additional feature engineering
            # Example: Calculate moving averages, RSI, etc.
            # data['moving_average'] = data['close'].rolling(window=window_size).mean()
            # ... Add other feature calculations as needed ...

            # Define target variable and feature set
            X = data.drop(['close', 'date'], axis=1) if 'date' in data.columns else data.drop(['close'], axis=1)
            y = data['close']

            # Apply the scaler
            scaler = self.get_scaler(scaler_type)
            X_scaled = scaler.fit_transform(X)

            # Time series split
            tscv = TimeSeriesSplit(n_splits=n_splits)
            for train_index, test_index in tscv.split(X_scaled):
                X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            return X_train, X_test, y_train, y_test

        except Exception as e:
            print(f"Error in data preprocessing: {e}")
            return None, None, None, None
    def get_scaler(self, scaler_type):
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'normalizer': Normalizer(),
            'maxabs': MaxAbsScaler()
        }
        return scalers.get(scaler_type, StandardScaler())
    
def run(self):
    for i in range(len(self.X_test)):
        current_data = self.X_test[i]
        prediction = self.model.predict([current_data])[0]

        # Define your trading strategy here
        # Example strategy: Buy if prediction > threshold, Sell if prediction < threshold
        threshold_buy = 0.5  # Placeholder value
        threshold_sell = -0.5  # Placeholder value
        trade_size = 100  # Placeholder value for trade size

        if prediction > threshold_buy:
            # Execute a buy trade
            trade = Trade(symbol="YourSymbol", size=trade_size, order_type="buy", price=current_data['price'])
            self.portfolio.execute_trade(trade, current_data['price'])
        elif prediction < threshold_sell:
            # Execute a sell trade
            trade = Trade(symbol="YourSymbol", size=-trade_size, order_type="sell", price=current_data['price'])
            self.portfolio.execute_trade(trade, current_data['price'])

        # Update portfolio value based on the latest market data
        # You need to provide the current market prices for each symbol in the portfolio
        current_market_prices = {"YourSymbol": current_data['price']}  # Placeholder
        self.portfolio.update_portfolio_value(current_market_prices)

    # Calculate and display trade analytics at the end
    trade_analytics = self.portfolio.calculate_trade_analytics()
    print("Trade Analytics:", trade_analytics)


    def plot_portfolio_performance(self):
        df = pd.DataFrame(self.portfolio.history)
        create_line_chart(df, x_column='date', y_column='portfolio_value', title="Portfolio Value Over Time")

    def plot_asset_performance(self):
        create_candlestick_chart(self.historical_data, title="Asset Price Performance")


    
# Example Usage
historical_data = 'path_to_your_data.csv'
model_backtest = MLModelBacktest(historical_data, 'path_to_your_saved_model.pkl', scaler_type, model_type, epochs, window_size)
model_backtest.run()
model_backtest.plot_portfolio_performance()
model_backtest.plot_asset_performance()
