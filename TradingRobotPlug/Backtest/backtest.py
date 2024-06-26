import aiohttp
import logging
import numpy as np
import joblib
from alpha_vantage.timeseries import TimeSeries
import configparser
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, MaxAbsScaler
from typing import Dict, Optional
from dataclasses import dataclass
import pandas as pd
from Data_processing.visualization import create_line_chart, create_candlestick_chart

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
        if trade.size == 0:
            print("Trade size is zero, no trade executed.")
            return

        trade_cost = trade.size * price
        if trade_cost > self.cash:
            print("Insufficient funds to execute trade.")
            return

        self.cash -= trade_cost
        self.positions[trade.symbol] = self.positions.get(trade.symbol, 0) + trade.size
        self.history.append({
            'symbol': trade.symbol, 
            'size': trade.size, 
            'price': price, 
            'portfolio_value': self.calculate_total_value(price, trade.symbol)
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

    def update_portfolio_value(self, current_prices):
        self.total_value = self.cash + sum(
            size * current_prices.get(symbol, 0) for symbol, size in self.positions.items())

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
    def calculate_total_value(self, current_prices):
        total_value = self.cash + sum(
            size * current_prices.get(symbol, 0) for symbol, size in self.positions.items()
        )
        return total_value

# MLModelBacktest Class
class MLModelBacktest:
    def __init__(self, data_file_path, model_path, scaler_type, window_size=5, n_splits=3):
        self.model = joblib.load(model_path)
        self.portfolio = Portfolio(initial_capital=10000)
        self.historical_data = pd.read_csv(data_file_path)
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocess_data(
            data_file_path, scaler_type, window_size, n_splits)
        
    def preprocess_data(self, data_file_path, scaler_type, window_size, n_splits):
        try:
            data = pd.read_csv(data_file_path)

            if not {'open', 'high', 'low', 'close', 'volume'}.issubset(data.columns):
                raise ValueError("DataFrame must contain OHLCV columns.")

            data = data.dropna()

            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data['day_of_week'] = data['date'].dt.dayofweek
                data['month'] = data['date'].dt.month
                data['year'] = data['date'].dt.year

            X = data.drop(['close', 'date'], axis=1) if 'date' in data.columns else data.drop(['close'], axis=1)
            y = data['close']

            scaler = self.get_scaler(scaler_type)
            X_scaled = scaler.fit_transform(X)

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

    def adjust_thresholds(self, prediction_confidence, trade_type):
        # Implement logic to adjust buy/sell thresholds based on prediction confidence and market conditions
        # Example implementation
        base_threshold = 0.5 if trade_type == 'buy' else -0.5
        adjusted_threshold = base_threshold * prediction_confidence
        return adjusted_threshold

    def calculate_trade_size(self, prediction_confidence):
        # Maximum risk per trade as a percentage of total portfolio value
        max_risk_per_trade = 0.02  # 2% of portfolio value

        # Adjust trade size based on prediction confidence and volatility
        # Higher confidence and lower volatility lead to larger trade sizes
        volatility_estimate = self.estimate_market_volatility()  # Implement this method
        risk_adjustment_factor = prediction_confidence / volatility_estimate

        # Calculate the dollar amount at risk
        amount_at_risk = self.portfolio.total_value * max_risk_per_trade * risk_adjustment_factor

        # Determine trade size. This could be further refined based on asset-specific characteristics
        trade_size = amount_at_risk / self.get_asset_price()  # Implement this method to get the current asset price

        return trade_size

    def get_asset_price(self, symbol, data_source="realtime", api_key=None, csv_file_path=None):
        """
        Get the current price of the specified asset.

        Args:
            symbol (str): The symbol or identifier of the asset.
            data_source (str): The data source to use for fetching the price.
                Options: "realtime" for real-time data, "historical" for historical data from Alpha Vantage,
                or "csv" for historical data from CSV files.
            api_key (str): Your Alpha Vantage API key (required for historical data from Alpha Vantage).
            csv_file_path (str): The path to the CSV file containing historical data (required for CSV data).

        Returns:
            float: The current price of the asset.
        """
        if data_source == "realtime":
            if not api_key:
                raise ValueError("API key is required for real-time data from Alpha Vantage.")
            
            try:
                ts = TimeSeries(key=api_key, output_format='pandas')
                data, meta_data = ts.get_quote_endpoint(symbol=symbol)
                if '05. price' in data:
                    return float(data['05. price'])
                else:
                    raise ValueError("Unable to fetch real-time price from Alpha Vantage.")
            except Exception as e:
                raise ValueError(f"Error fetching real-time price: {str(e)}")
        elif data_source == "historical":
            if not api_key:
                raise ValueError("API key is required for historical data from Alpha Vantage.")
            
            try:
                ts = TimeSeries(key=api_key, output_format='pandas')
                historical_data, meta_data = ts.get_daily(symbol=symbol, outputsize='compact')
                if not historical_data.empty:
                    # Get the latest price from historical data
                    latest_price = historical_data.iloc[-1]['4. close']
                    return float(latest_price)
                else:
                    raise ValueError(f"No historical data found for symbol {symbol}.")
            except Exception as e:
                raise ValueError(f"Error fetching historical price from Alpha Vantage: {str(e)}")
        elif data_source == "csv":
            if not csv_file_path:
                raise ValueError("CSV file path is required for historical data from CSV.")
            
            try:
                # Load historical data from CSV file
                historical_data = pd.read_csv(csv_file_path)
                # Filter data for the specified symbol and get the latest price
                symbol_data = historical_data[historical_data['symbol'] == symbol]
                if not symbol_data.empty:
                    latest_price = symbol_data.iloc[-1]['close']
                    return float(latest_price)
                else:
                    raise ValueError(f"No historical data found for symbol {symbol}.")
            except Exception as e:
                raise ValueError(f"Error fetching historical price from CSV: {str(e)}")
        else:
            raise ValueError("Invalid data_source. Use 'realtime', 'historical', or 'csv'.")


    def estimate_market_volatility(self, historical_prices):
        # Calculate daily returns from historical price data
        returns = np.diff(historical_prices) / historical_prices[:-1]

        # Calculate the standard deviation of daily returns as a measure of volatility
        volatility = np.std(returns)

        return volatility


    def run(self):

        for i in range(len(self.X_test)):
            current_data = self.X_test[i]
            prediction, prediction_confidence = self.model.predict([current_data])

            # Adjust thresholds and trade size based on prediction confidence and other factors
            dynamic_threshold_buy = self.adjust_thresholds(prediction_confidence, 'buy')
            dynamic_threshold_sell = self.adjust_thresholds(prediction_confidence, 'sell')
            trade_size = self.calculate_trade_size(prediction_confidence, current_data['symbol'])

            if prediction > dynamic_threshold_buy:
                trade = Trade(symbol=current_data['symbol'], size=trade_size, order_type="buy", price=current_data['price'])
                self.portfolio.execute_trade(trade, current_data['price'])
            elif prediction < dynamic_threshold_sell:
                trade = Trade(symbol=current_data['symbol'], size=-trade_size, order_type="sell", price=current_data['price'])
                self.portfolio.execute_trade(trade, current_data['price'])

            # Estimate market volatility using historical price data
            historical_prices = self.get_historical_prices(current_data['symbol'])  # Implement this method to fetch historical prices
            volatility_estimate = self.estimate_market_volatility(historical_prices)

    def get_historical_prices(self, symbol):
        # Implement logic to fetch historical prices for the specified asset
        # You can use data from a CSV file or an API, depending on your data source
        # Replace this with the actual logic for fetching historical prices
        historical_prices = []  # Placeholder list, replace with actual historical prices
        return historical_prices

    def plot_portfolio_performance(self):
        df = pd.DataFrame(self.portfolio.history)
        create_line_chart(df, x_column='date', y_column='portfolio_value', title="Portfolio Value Over Time")

    def plot_asset_performance(self):
        create_candlestick_chart(self.historical_data, title="Asset Price Performance")

    def update_trade_outcome_based_on_market_conditions(self, trade, take_profit, stop_loss):
        # Example implementation using simulated market prices
        # In a real scenario, this would involve real-time market data

        simulated_market_prices = [trade.price * (1 + np.random.normal(0, 0.02)) for _ in range(10)]  # Simulating 10 price movements

        for price in simulated_market_prices:
            if trade.order_type == 'buy':
                if price >= take_profit:
                    self.portfolio.update_trade_outcome(trade.symbol, take_profit)
                    break
                elif price <= stop_loss:
                    self.portfolio.update_trade_outcome(trade.symbol, stop_loss)
                    break
            elif trade.order_type == 'sell':
                if price <= take_profit:
                    self.portfolio.update_trade_outcome(trade.symbol, take_profit)
                    break
                elif price >= stop_loss:
                    self.portfolio.update_trade_outcome(trade.symbol, stop_loss)
                    break

# Custom exceptions for better error handling
class RealTimeDataError(Exception):
    def __init__(self, symbol: str, message: str):
        self.symbol = symbol
        self.message = message
        super().__init__(f"Real-time data error for symbol {symbol}: {message}")

class HistoricalDataError(Exception):
    def __init__(self, symbol: str, source: str, message: str):
        self.symbol = symbol
        self.source = source
        self.message = message
        super().__init__(f"Historical data error for symbol {symbol} from source {source}: {message}")

@dataclass
class RealTimeData:
    symbol: str
    price: float

@dataclass
class HistoricalData:
    symbol: str
    data: pd.DataFrame

class AssetDataManager:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger if logger else logging.getLogger(__name__)
        self.logger.setLevel(logging.ERROR)

    async def async_fetch_real_time_price(self, symbol: str) -> RealTimeData:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f'https://api.example.com/realtime/{symbol}') as response:
                    response_data = await response.json()

                    if 'price' in response_data:
                        return RealTimeData(symbol, float(response_data['price']))
                    else:
                        raise RealTimeDataError(f"Invalid response data for {symbol}: {response_data}")

        except aiohttp.ClientError as e:
            self.log_error(f"Error fetching real-time data for {symbol}: {str(e)}")
            raise RealTimeDataError(f"Error fetching real-time data for {symbol}: {str(e)}")

    def fetch_historical_data_csv(self, symbol: str, file_path: str) -> HistoricalData:
        try:
            # Implement logic to fetch historical data from a CSV file
            # Handle exceptions and errors gracefully
            historical_data = pd.read_csv(file_path)
            return HistoricalData(symbol, historical_data)

        except Exception as e:
            self.log_error(f"An unexpected error occurred while fetching historical CSV data for {symbol}: {str(e)}")
            raise HistoricalDataError(f"Error fetching historical CSV data for {symbol}: {str(e)}")

    def fetch_historical_data_alpha_vantage(self, symbol: str, api_key: str) -> HistoricalData:
        try:
            # Implement logic to fetch historical data from AlphaVantage API
            # Handle exceptions and errors gracefully
            # You can use the 'alpha_vantage' library for this purpose
            # Example: https://github.com/RomelTorres/alpha_vantage
            historical_data = {}  # Replace with actual historical data retrieval logic
            return HistoricalData(symbol, historical_data)

        except Exception as e:
            self.log_error(f"An unexpected error occurred while fetching AlphaVantage data for {symbol}: {str(e)}")
            raise HistoricalDataError(f"Error fetching AlphaVantage data for {symbol}: {str(e)}")

    def get_current_market_prices(self, symbols: list, use_csv: bool, csv_file_paths: Dict[str, str], api_key: Optional[str] = None) -> Dict[str, float]:
        current_prices = {}
        for symbol in symbols:
            try:
                if use_csv and symbol in csv_file_paths:
                    historical_data = self.fetch_historical_data_csv(symbol, csv_file_paths[symbol])
                elif api_key:
                    historical_data = self.fetch_historical_data_alpha_vantage(symbol, api_key)
                else:
                    raise HistoricalDataError("No data source specified for historical data")

                latest_price = historical_data.data['close'].iloc[-1]
                current_prices[symbol] = latest_price

            except (HistoricalDataError, RealTimeDataError) as e:
                # Handle data retrieval errors
                self.log_error(f"Error fetching data for {symbol}: {str(e)}")
                current_prices[symbol] = 0.0

        return current_prices

    def log_error(self, message: str):
        self.logger.error(message)
