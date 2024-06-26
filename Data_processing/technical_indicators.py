import pandas as pd
import numpy as np
from ta.trend import ADXIndicator
from ta.volatility import BollingerBands
from ta.momentum import StochasticOscillator
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.momentum import WilliamsRIndicator
from ta.momentum import ROCIndicator
from ta.volume import MFIIndicator  # Corrected import for Money Flow Index (MFI)
from ta.volatility import KeltnerChannel
from ta.trend import IchimokuIndicator
from ta.trend import MACD
from ta.trend import TRIXIndicator
from ta.volatility import BollingerBands
from ta.trend import PSARIndicator
from ta.volume import VolumeWeightedAveragePrice
from pandas_ta.volume import obv
from pandas_ta.momentum import ao
import talib
import time

class AverageTrueRange:
    def __init__(self, high, low, close, window_size):
        """
        Initializes the AverageTrueRange class with high, low, close prices and window size.

        :param high: pandas Series of high prices
        :param low: pandas Series of low prices
        :param close: pandas Series of close prices
        :param window_size: Integer, size of the rolling window to calculate ATR
        """
        if not (isinstance(high, pd.Series) and isinstance(low, pd.Series) and isinstance(close, pd.Series)):
            raise ValueError("High, low, and close must be pandas Series.")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        self.high = high
        self.low = low
        self.close = close
        self.window_size = window_size

    def _true_range(self):
        """
        Private method to calculate the true range.
        """
        high_low = self.high - self.low
        high_close_prev = abs(self.high - self.close.shift(1)).fillna(0)
        low_close_prev = abs(self.low - self.close.shift(1)).fillna(0)
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1)
        return tr.max(axis=1)

    def average_true_range(self):
        """
        Calculates the Average True Range (ATR) using a rolling window.
        """
        tr = self._true_range()
        atr = tr.rolling(window=self.window_size).mean()
        return atr

class TechnicalIndicators:
    @staticmethod
    def add_moving_average(df, window_size=10, user_defined_window=None, column='close', ma_type='SMA'):
        """
        Adds a moving average column to the DataFrame.

        :param df: pandas DataFrame containing price data
        :param window_size: Integer, default window size for moving average
        :param user_defined_window: Integer, user-defined window size, overrides default if provided
        :param column: String, the column name to calculate moving average on
        :param ma_type: String, type of moving average (e.g., 'SMA' or 'EMA')
        :return: DataFrame with the new moving average column added
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        if user_defined_window is not None:
            if not isinstance(user_defined_window, int) or user_defined_window <= 0:
                raise ValueError("User defined window size must be a positive integer.")
            window_size = user_defined_window
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        if ma_type.lower() == 'sma':
            df[f'SMA_{window_size}'] = df[column].rolling(window=window_size).mean()
        elif ma_type.lower() == 'ema':
            df[f'EMA_{window_size}'] = df[column].ewm(span=window_size, adjust=False).mean()
        else:
            raise ValueError(f"Moving average type '{ma_type}' is not supported.")

        return df

    @staticmethod
    def add_bollinger_bands(df, window_size=10, std_multiplier=2, user_defined_window=None):
        """
        Adds Bollinger Bands to the DataFrame.

        :param df: pandas DataFrame with price data
        :param window_size: Integer, default window size for Bollinger Bands
        :param std_multiplier: Integer, standard deviation multiplier for Bollinger Bands
        :param user_defined_window: Integer, user-defined window size, overrides default if provided
        :return: DataFrame with Bollinger Bands columns added
        """
        if user_defined_window is not None:
            window_size = user_defined_window
        if 'close' not in df.columns:
            raise ValueError("Column 'close' not found in DataFrame")

        rolling_mean = df['close'].rolling(window=window_size, min_periods=1).mean()
        rolling_std = df['close'].rolling(window=window_size, min_periods=1).std().fillna(0)

        df['Bollinger_High'] = rolling_mean + (rolling_std * std_multiplier)
        df['Bollinger_Low'] = rolling_mean - (rolling_std * std_multiplier)
        df['Bollinger_Mid'] = rolling_mean

        return df

    @staticmethod
    def add_exponential_moving_average(df, column='close', window_size=10):
        """
        Adds an Exponential Moving Average (EMA) column to the DataFrame.

        :param df: pandas DataFrame containing price data
        :param column: String, the column name to calculate EMA on
        :param window_size: Integer, window size for EMA calculation
        :return: DataFrame with the EMA column added
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        ema = df[column].ewm(span=window_size, adjust=False).mean()
        df[f'EMA_{window_size}'] = ema

        return df

    @staticmethod
    def add_stochastic_oscillator(df, window_size=14, user_defined_window=None):
        """
        Adds Stochastic Oscillator and its signal line to the DataFrame.

        :param df: pandas DataFrame containing high, low, and close price data
        :param window_size: Integer, default window size for the Stochastic Oscillator
        :param user_defined_window: Integer, user-defined window size, overrides default if provided
        :return: DataFrame with Stochastic Oscillator and Signal columns added
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        for column in ['low', 'high', 'close']:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")
        if user_defined_window is not None:
            if not isinstance(user_defined_window, int) or user_defined_window <= 0:
                raise ValueError("User defined window size must be a positive integer.")
            window_size = user_defined_window
        elif not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        lowest_low = df['low'].rolling(window=window_size, min_periods=1).min()
        highest_high = df['high'].rolling(window=window_size, min_periods=1).max()

        # Handling division by zero
        denominator = highest_high - lowest_low
        denominator[denominator == 0] = 1

        df['Stochastic'] = 100 * ((df['close'] - lowest_low) / denominator)
        df['Stochastic_Signal'] = df['Stochastic'].rolling(window=3, min_periods=1).mean()

        return df

    @staticmethod
    def calculate_macd_components(df, fast_period=12, slow_period=26, signal_period=9, price_column='close'):
        """
        Calculate MACD components for a given DataFrame.

        Args:
            df (DataFrame): Stock price data.
            fast_period (int): The fast period for EMA calculation, must be non-negative.
            slow_period (int): The slow period for EMA calculation, must be non-negative.
            signal_period (int): The signal period for EMA calculation, must be non-negative.
            price_column (str): Column name for price data in df.

        Returns:
            DataFrame: DataFrame with MACD, MACD Signal, MACD Histogram, and MACD Histogram Signal components added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input 'df' must be a pandas DataFrame.")
        if not all(isinstance(x, int) and x >= 0 for x in [fast_period, slow_period, signal_period]):
            raise ValueError("Period parameters must be non-negative integers.")
        if price_column not in df.columns:
            raise ValueError(f"'{price_column}' column not found in DataFrame.")

        # Calculate MACD components
        fast_ema = df[price_column].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df[price_column].ewm(span=slow_period, adjust=False).mean()
        df['MACD'] = fast_ema - slow_ema
        df['MACD_Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        df['MACD_Hist_Signal'] = df['MACD_Hist'].ewm(span=signal_period, adjust=False).mean()

        return df

    @staticmethod
    def add_average_true_range(df, window_size=14, user_defined_window=None, high_col='high', low_col='low', close_col='close'):
        """
        Adds the Average True Range (ATR) column to the DataFrame.

        Args:
            df (DataFrame): DataFrame with stock price data.
            window_size (int): Default window size for ATR calculation.
            user_defined_window (int): Optional, user-defined window size for ATR.
            high_col (str): Column name for high prices.
            low_col (str): Column name for low prices.
            close_col (str): Column name for close prices.

        Returns:
            DataFrame: Modified DataFrame with the ATR column added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        for col in [high_col, low_col, close_col]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

        window = user_defined_window if user_defined_window is not None else window_size
        if not isinstance(window, int) or window <= 0:
            raise ValueError("Window size must be a positive integer.")

        try:
            indicator = AverageTrueRange(df[high_col], df[low_col], df[close_col], window)
            df['ATR'] = indicator.average_true_range()
            return df
        except Exception as e:
            raise RuntimeError(f"An error occurred while calculating ATR: {e}")

    # Section 2: Other Oscillator Indicators

    @staticmethod
    def add_relative_strength_index(df, window=14, user_defined_window=None, calculation_type="default"):
        """
        Adds the Relative Strength Index (RSI) to the DataFrame.

        Args:
            df (DataFrame): DataFrame containing stock price data.
            window (int): Default window size for RSI calculation.
            user_defined_window (int): Optional, user-defined window size for RSI.
            calculation_type (str): Type of RSI calculation ('default' or 'custom').

        Returns:
            DataFrame: Modified DataFrame with the RSI column added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if 'close' not in df.columns:
            raise ValueError("Column 'close' not found in DataFrame")

        window_size = user_defined_window if user_defined_window is not None else window
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        delta = df['close'].diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        if calculation_type == "custom":
            avg_gain = gain.rolling(window=window_size).mean()
            avg_loss = loss.rolling(window=window_size).mean()
        else:  # Default calculation
            avg_gain = gain.rolling(window=window_size, min_periods=1).mean()
            avg_loss = loss.rolling(window=window_size, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        df['RSI'] = rsi.fillna(0)
        return df

    @staticmethod
    def add_commodity_channel_index(df, window=20, user_defined_window=None):
        """
        Adds the Commodity Channel Index (CCI) to the DataFrame.

        Args:
            df (DataFrame): DataFrame containing stock price data.
            window (int): Default window size for CCI calculation.
            user_defined_window (int): Optional, user-defined window size for CCI.

        Returns:
            DataFrame: Modified DataFrame with the CCI column added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        for column in ['high', 'low', 'close']:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")

        window_size = user_defined_window if user_defined_window is not None else window
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        typical_price = (df['high'] + df['low'] + df['close']) / 3
        mean_deviation = lambda x: abs(x - x.mean()).mean()
        cci = (typical_price - typical_price.rolling(window=window_size).mean()) / (0.015 * typical_price.rolling(window=window_size).apply(mean_deviation))

        df['CCI'] = cci.fillna(0)
        return df

    @staticmethod
    def add_williams_r(df, window=14, user_defined_window=None):
        """
        Adds the Williams %R indicator to the DataFrame.

        Args:
            df (DataFrame): DataFrame containing stock price data.
            window (int): Default window size for Williams %R calculation.
            user_defined_window (int): Optional, user-defined window size for Williams %R.

        Returns:
            DataFrame: Modified DataFrame with the Williams %R column added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        for column in ['high', 'low', 'close']:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")

        window_size = user_defined_window if user_defined_window is not None else window
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        highest_high = df['high'].rolling(window=window_size, min_periods=1).max()
        lowest_low = df['low'].rolling(window=window_size, min_periods=1).min()

        # Handling division by zero
        denominator = highest_high - lowest_low
        denominator[denominator == 0] = 1

        df['Williams_R'] = -100 * (highest_high - df['close']) / denominator
        df['Williams_R'] = df['Williams_R'].fillna(0)
        return df

    @staticmethod
    def add_rate_of_change(df, window=10, user_defined_window=None):
        """
        Adds the Rate of Change (ROC) indicator to the DataFrame.

        Args:
            df (DataFrame): DataFrame containing stock price data.
            window (int): Default window size for ROC calculation.
            user_defined_window (int): Optional, user-defined window size for ROC.

        Returns:
            DataFrame: Modified DataFrame with the ROC column added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if 'close' not in df.columns:
            raise ValueError("Column 'close' not found in DataFrame")

        window_size = user_defined_window if user_defined_window is not None else window
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        # Calculating ROC
        shifted_close = df['close'].shift(window_size)
        df['ROC'] = ((df['close'] - shifted_close) / shifted_close) * 100
        df['ROC'] = df['ROC'].fillna(0)

        return df

    @staticmethod
    def add_money_flow_index(df, window=14, user_defined_window=None):
        """
        Adds the Money Flow Index (MFI) to the DataFrame.

        Args:
            df (DataFrame): DataFrame containing stock price data, including volume.
            window (int): Default window size for MFI calculation.
            user_defined_window (int): Optional, user-defined window size for MFI.

        Returns:
            DataFrame: Modified DataFrame with the MFI column added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        for column in ['high', 'low', 'close', 'volume']:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")

        window_size = user_defined_window if user_defined_window is not None else window
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        positive_flow = money_flow.where(df['close'] > df['close'].shift(1), 0).rolling(window=window_size).sum()
        negative_flow = money_flow.where(df['close'] < df['close'].shift(1), 0).rolling(window=window_size).sum()

        # Handling division by zero
        mfi = 100 - (100 / (1 + positive_flow / negative_flow.replace(0, 1)))

        df['MFI'] = mfi.fillna(0)
        return df

    # Section 3: Volatility Indicators

    @staticmethod
    def add_standard_pivot_points(df, high='high', low='low', close='close'):
        """
        Adds Standard Pivot Points and associated support/resistance levels to the DataFrame.

        Pivot Point is a technical analysis indicator used to determine the overall trend of the market 
        over different time frames. The pivot itself is the average of the high, low, and closing prices 
        from the previous trading day. Support and resistance levels are then calculated off the pivot.

        Args:
            df (DataFrame): DataFrame containing stock price data.
            high (str): Column name for high prices.
            low (str): Column name for low prices.
            close (str): Column name for close prices.

        Returns:
            DataFrame: Modified DataFrame with Pivot Points and support/resistance levels added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if not all(column in df.columns for column in [high, low, close]):
            raise ValueError("DataFrame must contain specified high, low, and close columns")

        # Calculate Pivot Points
        pivot = (df[high].shift(1) + df[low].shift(1) + df[close].shift(1)) / 3
        df['Pivot_Point'] = pivot
        df['R1'] = 2 * pivot - df[low].shift(1)   # First Resistance Level
        df['S1'] = 2 * pivot - df[high].shift(1)  # First Support Level
        df['R2'] = pivot + (df[high].shift(1) - df[low].shift(1))  # Second Resistance Level
        df['S2'] = pivot - (df[high].shift(1) - df[low].shift(1))  # Second Support Level
        df['R3'] = df[high].shift(1) + 2 * (pivot - df[low].shift(1))  # Third Resistance Level
        df['S3'] = df[low].shift(1) - 2 * (df[high].shift(1) - pivot)  # Third Support Level

        return df

    @staticmethod
    def add_standard_deviation(df, window_size=20, user_defined_window=None):
        """
        Adds a rolling standard deviation to the DataFrame based on the 'close' prices.

        Args:
            df (DataFrame): DataFrame containing stock price data.
            window_size (int): Default window size for calculating standard deviation.
            user_defined_window (int): Optional, user-defined window size for standard deviation.

        Returns:
            DataFrame: Modified DataFrame with the rolling standard deviation column added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if 'close' not in df.columns:
            raise ValueError("Column 'close' not found in DataFrame")

        window = user_defined_window if user_defined_window is not None else window_size
        if not isinstance(window, int) or window <= 0:
            raise ValueError("Window size must be a positive integer.")

        df['Standard_Deviation'] = df['close'].rolling(window=window).std().fillna(0)
        return df

    @staticmethod
    def add_historical_volatility(df, window=20, user_defined_window=None):
        """
        Adds historical volatility to the DataFrame, calculated as the standard deviation
        of the logarithmic returns of closing prices.

        Args:
            df (pd.DataFrame): DataFrame containing the 'close' prices.
            window (int): The window size for calculating volatility. Defaults to 20.
            user_defined_window (int, optional): User-defined window size. If provided, it overrides the default.

        Returns:
            pd.DataFrame: DataFrame with the new 'Historical_Volatility' column.
        """

        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column")

        window_size = user_defined_window if user_defined_window is not None else window
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        # Calculate log returns, handling division by zero or negative values
        log_return = np.log(df['close'] / df['close'].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0)

        # Calculate and add historical volatility column
        df['Historical_Volatility'] = log_return.rolling(window=window_size).std() * np.sqrt(window_size)

        return df

    @staticmethod
    def add_chandelier_exit(df, window=22, multiplier=3, user_defined_window=None, user_defined_multiplier=None):
        """
        Adds Chandelier Exit indicators to the DataFrame.

        Args:
        df (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' prices.
        window (int, optional): The window size for calculating the indicator. Defaults to 22.
        multiplier (float, optional): Multiplier for the ATR value. Defaults to 3.
        user_defined_window (int, optional): User defined window size. If provided, it overrides the default.
        user_defined_multiplier (float, optional): User defined multiplier. If provided, it overrides the default.

        Returns:
        pd.DataFrame: DataFrame with new 'Chandelier_Exit_Long' column.
        """

        # Validate if required columns exist in the DataFrame
        required_columns = ['high', 'low', 'close']
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {', '.join(required_columns)}")

        # Override window and multiplier if user-defined values are provided
        if user_defined_window is not None:
            window = user_defined_window
        if user_defined_multiplier is not None:
            multiplier = user_defined_multiplier

        # Calculate the highest high and ATR
        highest_high = df['high'].rolling(window=window).max()
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=window)

        # Calculate and add Chandelier Exit Long column
        df['Chandelier_Exit_Long'] = highest_high - multiplier * atr

        return df

    @staticmethod
    def add_keltner_channel(df, window=20, multiplier=2, user_defined_window=None, user_defined_multiplier=None):
        """
        Adds Keltner Channel to the DataFrame.

        The Keltner Channel is a volatility-based trading indicator. It consists of three lines:
        a middle line (MA of the closing price), an upper line (MA + multiplier * ATR),
        and a lower line (MA - multiplier * ATR).

        Args:
            df (pd.DataFrame): DataFrame with 'high', 'low', and 'close' prices.
            window (int): The window size for the moving average and ATR. Defaults to 20.
            multiplier (float): Multiplier for the ATR. Defaults to 2.
            user_defined_window (int, optional): User-defined window size. Overrides default if provided.
            user_defined_multiplier (float, optional): User-defined multiplier. Overrides default if provided.

        Returns:
            pd.DataFrame: DataFrame with Keltner Channel columns added.
        """

        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        required_columns = ['high', 'low', 'close']
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {', '.join(required_columns)}")

        window_size = user_defined_window if user_defined_window is not None else window
        atr_multiplier = user_defined_multiplier if user_defined_multiplier is not None else multiplier

        # Calculate the Keltner Channel
        ma = df['close'].rolling(window=window_size).mean()
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=window_size)
        df['Keltner_Channel_High'] = ma + (atr_multiplier * atr)
        df['Keltner_Channel_Low'] = ma - (atr_multiplier * atr)
        df['Keltner_Channel_Mid'] = ma

        return df

    @staticmethod
    def add_moving_average_envelope(df, window_size=10, percentage=0.025, user_defined_window=None, user_defined_percentage=None):
        """
        Adds Moving Average Envelope to the DataFrame.

        The Moving Average Envelope consists of two lines, which are calculated as a percentage above 
        and below a moving average (SMA) of the closing price.

        Args:
            df (pd.DataFrame): DataFrame containing stock price data.
            window_size (int): The window size for calculating the SMA. Defaults to 10.
            percentage (float): The percentage above and below the SMA for the envelope. Defaults to 0.025 (2.5%).
            user_defined_window (int, optional): User-defined window size for SMA. Overrides default if provided.
            user_defined_percentage (float, optional): User-defined percentage for the envelope. Overrides default if provided.

        Returns:
            pd.DataFrame: DataFrame with MAE upper and lower bounds added.
        """

        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if 'close' not in df.columns:
            raise ValueError("Column 'close' not found in DataFrame")

        window = user_defined_window if user_defined_window is not None else window_size
        envelope_percentage = user_defined_percentage if user_defined_percentage is not None else percentage

        # Validate window and percentage
        if not isinstance(window, int) or window <= 0:
            raise ValueError("Window size must be a positive integer.")
        if not isinstance(envelope_percentage, float) or not (0 <= envelope_percentage <= 1):
            raise ValueError("Percentage must be a float between 0 and 1.")

        # Calculate Moving Average Envelope
        SMA = df['close'].rolling(window=window).mean()
        df['MAE_Upper'] = SMA * (1 + envelope_percentage)
        df['MAE_Lower'] = SMA * (1 - envelope_percentage)

        return df

    @staticmethod
    def add_adx(df, window=14, user_defined_window=None):
        """
        Adds the Average Directional Index (ADX) to the DataFrame.

        The ADX is a technical analysis indicator used to quantify trend strength. The higher the ADX value,
        the stronger the trend.

        Args:
            df (pd.DataFrame): DataFrame with 'high', 'low', and 'close' prices.
            window (int): The window size for calculating ADX. Defaults to 14.
            user_defined_window (int, optional): User-defined window size for ADX. Overrides default if provided.

        Returns:
            pd.DataFrame: Modified DataFrame with the ADX column added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        for column in ['high', 'low', 'close']:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")

        window_size = user_defined_window if user_defined_window is not None else window
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        if len(df) >= window_size:
            adx_indicator = ADXIndicator(df['high'], df['low'], df['close'], window_size, fillna=True)
            df['ADX'] = adx_indicator.adx()
        else:
            df['ADX'] = pd.NA  # Filling with pandas NA for better handling

        return df

    @staticmethod
    def add_ichimoku_cloud(df, user_defined_values=None):
        """
        Adds Ichimoku Cloud indicators to the DataFrame.

        Ichimoku Cloud consists of five lines (Conversion Line, Base Line, Leading Span A,
        Leading Span B, and Lagging Span) and is used to identify trends, supports, and resistances.

        Args:
            df (pd.DataFrame): DataFrame with 'high', 'low', and 'close' prices.
            user_defined_values (tuple, optional): User-defined window sizes (nine_window, twenty_six_window, fifty_two_window).

        Returns:
            pd.DataFrame: Modified DataFrame with Ichimoku Cloud components added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        for column in ['high', 'low', 'close']:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")

        if user_defined_values is not None:
            if not (isinstance(user_defined_values, tuple) and len(user_defined_values) == 3):
                raise ValueError("User defined values must be a tuple of three integers.")
            nine_window, twenty_six_window, fifty_two_window = user_defined_values
        else:
            nine_window, twenty_six_window, fifty_two_window = 9, 26, 52

        def calculate_line(window):
            period_high = df['high'].rolling(window=window).max()
            period_low = df['low'].rolling(window=window).min()
            return (period_high + period_low) / 2

        df['Ichimoku_Conversion_Line'] = calculate_line(nine_window)
        df['Ichimoku_Base_Line'] = calculate_line(twenty_six_window)
        df['Ichimoku_Leading_Span_A'] = ((df['Ichimoku_Conversion_Line'] + df['Ichimoku_Base_Line']) / 2).shift(twenty_six_window)
        df['Ichimoku_Leading_Span_B'] = calculate_line(fifty_two_window).shift(twenty_six_window)
        df['Ichimoku_Lagging_Span'] = df['close'].shift(-twenty_six_window)

        return df

    @staticmethod
    def add_parabolic_sar(df, step=0.02, max_step=0.2):
        """
        Adds the Parabolic SAR (Stop and Reverse) to the DataFrame.

        The Parabolic SAR is a price-and-time-based trading system. It provides potential stop-loss levels
        and indicates the direction of the trend.

        Args:
            df (pd.DataFrame): DataFrame with 'high', 'low', and 'close' prices.
            step (float): The step increment. Defaults to 0.02.
            max_step (float): The maximum step. Defaults to 0.2.

        Returns:
            pd.DataFrame: Modified DataFrame with the Parabolic SAR column added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if not all(column in df.columns for column in ['high', 'low', 'close']):
            raise ValueError("DataFrame must contain 'high', 'low', and 'close' columns")

        # Initialize Parabolic SAR values
        psar = df['close'][0]
        psar_high = df['high'][0]
        psar_low = df['low'][0]
        bullish = True
        af = step

        # Initialize a Series to store PSAR values
        psar_values = pd.Series(index=df.index)
        psar_values.iloc[0] = psar

        for i in range(1, len(df)):
            prior_psar = psar
            prior_psar_high = psar_high
            prior_psar_low = psar_low

            if bullish:
                psar = prior_psar + af * (prior_psar_high - prior_psar)
                psar_high = max(prior_psar_high, df['high'].iloc[i])
                if df['low'].iloc[i] < psar:
                    bullish = False
                    psar = prior_psar_high
                    af = step
            else:
                psar = prior_psar + af * (prior_psar_low - prior_psar)
                psar_low = min(prior_psar_low, df['low'].iloc[i])
                if df['high'].iloc[i] > psar:
                    bullish = True
                    psar = prior_psar_low
                    af = step

            if bullish:
                psar = min(psar, df['low'].iloc[i - 1])
            else:
                psar = max(psar, df['high'].iloc[i - 1])

            if (bullish and df['high'].iloc[i] > psar_high) or (not bullish and df['low'].iloc[i] < psar_low):
                af = min(af + step, max_step)

            psar_values.iloc[i] = psar

        df['PSAR'] = psar_values
        return df

    @staticmethod
    def determine_threshold(df, dynamic=True, fixed_threshold=2.0, vol_window=20):
        """
        Determines a threshold based on historical volatility or a fixed value.

        Args:
            df (pd.DataFrame): DataFrame containing stock price data.
            dynamic (bool): If True, calculates a dynamic threshold based on historical volatility. 
                            If False, uses a fixed threshold. Defaults to True.
            fixed_threshold (float): Fixed threshold value to be used if dynamic is False. Defaults to 2.0.
            vol_window (int): The window size for calculating historical volatility if dynamic is True. Defaults to 20.

        Returns:
            pd.Series: Series containing the calculated threshold for each row in the DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")

        if dynamic:
            TechnicalIndicators.add_historical_volatility(df, window=vol_window)
            volatility = df['Historical_Volatility'].fillna(0)
            threshold = volatility * fixed_threshold
        else:
            threshold = pd.Series([fixed_threshold] * len(df), index=df.index)

        return threshold

    @staticmethod
    def add_zigzag_indicator(df, lookback=5, dynamic_threshold=True, fixed_threshold=2.0, vol_window=20):
        threshold = TechnicalIndicators.determine_threshold(df, dynamic=dynamic_threshold, fixed_threshold=fixed_threshold, vol_window=vol_window)

        # Debugging: Print the type and sample of the threshold
        print("Type of threshold:", type(threshold))
        print("Sample of threshold:", threshold.head())

        df['ZigZag'] = np.nan

        for i in range(lookback, len(df)):
            current_close = df['close'].iloc[i]
            previous_close = df['close'].iloc[i - lookback]

            # Check if the current_threshold is a valid DataFrame or Series
            current_threshold = threshold['Historical_Volatility'].iloc[i] if isinstance(threshold, pd.DataFrame) else threshold.iloc[i]

            # Debugging: Print the types and values of variables used in the comparison
            print(f"Index: {i}, Current Close: {current_close}, Previous Close: {previous_close}, Current Threshold: {current_threshold}")

            if pd.notna(current_threshold) and abs(current_close - previous_close) > current_threshold:
                df.loc[i, 'ZigZag'] = current_close

        return df

    # Section 6: Volume Indicators

    @staticmethod
    def add_on_balance_volume(df, user_defined_window=None):
        """
        Adds On-Balance Volume (OBV) to the DataFrame.

        OBV is a technical trading momentum indicator that uses volume flow to predict changes in stock price.

        Args:
            df (pd.DataFrame): DataFrame containing 'volume' and 'close' price data.
            user_defined_window (int, optional): User-defined window size. If provided, overrides default window size.

        Returns:
            pd.DataFrame: Modified DataFrame with the OBV column added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if not all(column in df.columns for column in ['volume', 'close']):
            raise ValueError("DataFrame must contain 'volume' and 'close' columns")

        window_size = user_defined_window if user_defined_window is not None else 14
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        obv_change = df['volume'] * np.sign(df['close'].diff()).fillna(0)
        df['OBV'] = obv_change.cumsum()

        return df

    @staticmethod
    def add_vwap(df, user_defined_window=None, price_type='typical', adjust_for_splits=False):
        """
        Adds Volume Weighted Average Price (VWAP) to the DataFrame, using vectorized operations for efficiency.

        VWAP is calculated as the sum of price multiplied by volume, divided by the total volume.

        Args:
            df (pd.DataFrame): DataFrame containing 'high', 'low', 'close', and 'volume' data.
            user_defined_window (int, optional): Window size for VWAP calculation. Defaults to length of DataFrame if None.
            price_type (str): Type of price to use ('typical', 'weighted_close'). 'typical' uses (high+low+close)/3.
            adjust_for_splits (bool): If True, adjusts for stock splits and dividends (requires 'split_coefficient' column).

        Returns:
            pd.DataFrame: Modified DataFrame with the VWAP column added.
        """
        start_time = time.time()

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        required_columns = ['high', 'low', 'close', 'volume']
        if adjust_for_splits and 'split_coefficient' not in df.columns:
            raise ValueError("DataFrame must contain a 'split_coefficient' column for split adjustments.")
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {', '.join(required_columns)}")

        window_size = len(df) if user_defined_window is None else user_defined_window
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        if price_type == 'typical':
            prices = (df['high'] + df['low'] + df['close']) / 3
        elif price_type == 'weighted_close':
            prices = (df['high'] + df['low'] + 2 * df['close']) / 4
        else:
            raise ValueError("Invalid price type specified.")

        if adjust_for_splits:
            adjusted_volume = df['volume'] / df['split_coefficient']
        else:
            adjusted_volume = df['volume']

        vwap = (prices * adjusted_volume).rolling(window=window_size).sum() / adjusted_volume.rolling(window=window_size).sum()
        df['VWAP'] = vwap

        execution_time = time.time() - start_time
        print(f"VWAP calculation completed in {execution_time:.2f} seconds.")

        return df

    @staticmethod
    def add_accumulation_distribution_line(df, user_defined_window=None):
        """
        Adds the Accumulation/Distribution Line (ADL) to the DataFrame.

        The ADL is a volume-based indicator designed to measure the cumulative flow of money into and out of a security.

        Args:
            df (pd.DataFrame): DataFrame with 'close', 'low', 'high', and 'volume' data.
            user_defined_window (int, optional): User-defined window size. Defaults to 20 if None.

        Returns:
            pd.DataFrame: Modified DataFrame with the ADL column added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        required_columns = ['close', 'low', 'high', 'volume']
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {', '.join(required_columns)}")

        window_size = user_defined_window if user_defined_window is not None else 20
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        high_low_diff = df['high'] - df['low']
        high_low_diff.replace(to_replace=0, method='ffill', inplace=True)  # Prevent division by zero

        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / high_low_diff
        clv.fillna(0, inplace=True)  # Handle NaN values
        df['ADL'] = (clv * df['volume']).cumsum()

        return df

    @staticmethod   
    def add_chaikin_money_flow(df, user_defined_window=None):
        if user_defined_window is not None:
            window = user_defined_window
        else:
            window = 14  # Default value
        
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv.fillna(0, inplace=True)
        money_flow_volume = clv * df['volume']
        df['CMF'] = money_flow_volume.rolling(window=window).sum() / df['volume'].rolling(window=window).sum()
        return df


    # Add more volume indicators as needed...

# Section 7: Other Indicators

    @staticmethod
    def add_trix(df, span=15, signal_line_span=9):
        """
        Add TRIX and its signal line to the DataFrame.

        TRIX is a momentum indicator that shows the percentage change in a triple exponentially smoothed moving average.
        The signal line is an EMA of the TRIX.

        Args:
            df (DataFrame): DataFrame with a 'close' column.
            span (int): The span for calculating TRIX. Default is 15.
            signal_line_span (int): The span for calculating the signal line. Default is 9.

        Returns:
            pd.DataFrame: DataFrame with TRIX and signal line columns added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column")
        if not isinstance(span, int) or span <= 0 or not isinstance(signal_line_span, int) or signal_line_span <= 0:
            raise ValueError("Span and signal line span must be positive integers.")

        # Calculate the first EMA
        ema1 = df['close'].ewm(span=span, adjust=False).mean()

        # Calculate the second EMA
        ema2 = ema1.ewm(span=span, adjust=False).mean()

        # Calculate the third EMA
        ema3 = ema2.ewm(span=span, adjust=False).mean()

        # Calculate TRIX
        df['TRIX'] = 100 * (ema3.pct_change())

        # Calculate the signal line (EMA of TRIX)
        df['TRIX_signal'] = df['TRIX'].ewm(span=signal_line_span, adjust=False).mean()

        return df

# Section 8: Custom Indicators

    @staticmethod
    def add_custom_indicator(df, indicator_name, indicator_function, *args, **kwargs):
        """
        Adds a custom indicator to the DataFrame using a user-defined function.

        Args:
            df (pd.DataFrame): DataFrame to which the indicator will be added.
            indicator_name (str): Name of the new indicator column to be added.
            indicator_function (callable): Function that computes the indicator.
            *args: Variable length argument list for the indicator function.
            **kwargs: Arbitrary keyword arguments for the indicator function.

        Returns:
            pd.DataFrame: Modified DataFrame with the new indicator column added.

        Raises:
            ValueError: If 'df' is not a pandas DataFrame or 'indicator_function' is not callable.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if not callable(indicator_function):
            raise ValueError("'indicator_function' must be a callable function.")

        try:
            df[indicator_name] = indicator_function(df, *args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Error in executing the custom indicator function: {e}")

        return df

    @staticmethod
    def add_fibonacci_retracement_levels(df, lookback_period=120):
        """
        Adds dynamic Fibonacci retracement levels to the DataFrame based on a lookback period.

        Args:
            df (pd.DataFrame): DataFrame with 'high' and 'low' columns.
            lookback_period (int): The number of periods to look back for peak and trough detection. Defaults to 120.

        Returns:
            pd.DataFrame: DataFrame with Fibonacci retracement level columns added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if 'high' not in df.columns or 'low' not in df.columns:
            raise ValueError("DataFrame must contain 'high' and 'low' columns")
        if lookback_period <= 0 or not isinstance(lookback_period, int):
            raise ValueError("Lookback period must be a positive integer.")

        # Identify local peak and trough within the lookback period
        rolling_max = df['high'].rolling(window=lookback_period, min_periods=1).max()
        rolling_min = df['low'].rolling(window=lookback_period, min_periods=1).min()

        # Determine if the market is trending upwards or downwards
        is_upward_trend = rolling_max.diff().fillna(0) >= 0

        # Calculate Fibonacci levels
        fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 1]
        for level in fib_levels:
            df[f'Fib_{int(level*100)}%'] = np.where(
                is_upward_trend,
                rolling_max - (rolling_max - rolling_min) * level,
                rolling_min + (rolling_max - rolling_min) * level
            )

        return df
    
    @staticmethod
    def add_awesome_oscillator(df, short_window=5, long_window=34):
        """
        Adds the Awesome Oscillator (AO) to the DataFrame.

        The AO is a market momentum indicator that is used to gauge whether bullish or bearish forces are currently driving the market.
        It is calculated as the difference between a short period and a long period simple moving average (SMA) of the mid price (average of high and low).

        Args:
            df (pd.DataFrame): DataFrame containing 'high' and 'low' data.
            short_window (int): Window size for the shorter period SMA. Defaults to 5.
            long_window (int): Window size for the longer period SMA. Defaults to 34.

        Returns:
            pd.DataFrame: Modified DataFrame with the Awesome Oscillator column added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        required_columns = ['high', 'low']
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {', '.join(required_columns)}")

        if not isinstance(short_window, int) or short_window <= 0 or not isinstance(long_window, int) or long_window <= 0:
            raise ValueError("Window sizes must be positive integers.")

        mid_price = (df['high'] + df['low']) / 2
        df['Awesome_Oscillator'] = mid_price.rolling(window=short_window).mean() - mid_price.rolling(window=long_window).mean()

        return df

    @staticmethod
    def add_volume_oscillator(df, short_window=12, long_window=26):
        """
        Adds the Volume Oscillator to the DataFrame.

        The Volume Oscillator measures the difference between two volume moving averages, 
        highlighting the increasing or decreasing volume trends.

        Args:
            df (pd.DataFrame): DataFrame containing 'volume' data.
            short_window (int): Window size for the shorter volume moving average. Defaults to 12.
            long_window (int): Window size for the longer volume moving average. Defaults to 26.

        Returns:
            pd.DataFrame: Modified DataFrame with the Volume Oscillator column added.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        if 'volume' not in df.columns:
            raise ValueError("DataFrame must contain a 'volume' column")

        if not isinstance(short_window, int) or not isinstance(long_window, int) or short_window <= 0 or long_window <= 0:
            raise ValueError("Window sizes must be positive integers.")
        if short_window >= long_window:
            raise ValueError("Short window size must be less than long window size.")

        short_vol_ema = df['volume'].ewm(span=short_window, adjust=False).mean()
        long_vol_ema = df['volume'].ewm(span=long_window, adjust=False).mean()
        df['Volume_Oscillator'] = short_vol_ema - long_vol_ema

        return df

# End of the Technical Indicators module

# Example of Applying Selected Features/Indicators in process_data function

# def process_data(file_path_entry, features_listbox, status_output, log_text):
    # ... [initialization and setup code] ...

    # Apply selected features/indicators
    # for feature in selected_features:
    #     if feature == "Simple Moving Average (SMA)":
    #         df = TechnicalIndicators.add_moving_average(df)
    #     elif feature == "Exponential Moving Average (EMA)":
    #         df = TechnicalIndicators.add_exponential_moving_average(df)
    #     elif feature == "Bollinger Bands":
    #         df = TechnicalIndicators.add_bollinger_bands(df)
    #     elif feature == "Stochastic Oscillator":
    #         df = TechnicalIndicators.add_stochastic_oscillator(df)
    #     elif feature == "MACD":
    #         df = TechnicalIndicators.calculate_macd_components(df)
    #     elif feature == "Average True Range (ATR)":
    #         df = TechnicalIndicators.add_average_true_range(df)
    #     elif feature == "Relative Strength Index (RSI)":
    #         df = TechnicalIndicators.add_relative_strength_index(df)
    #     elif feature == "Commodity Channel Index (CCI)":
    #         df = TechnicalIndicators.add_commodity_channel_index(df)
    #     elif feature == "Williams %R":
    #         df = TechnicalIndicators.add_williams_r(df)
    #     elif feature == "Rate of Change (ROC)":
    #         df = TechnicalIndicators.add_rate_of_change(df)
    #     elif feature == "Money Flow Index (MFI)":
    #         df = TechnicalIndicators.add_money_flow_index(df)
    #     elif feature == "Standard Deviation":
    #         df = TechnicalIndicators.add_standard_deviation(df)
    #     elif feature == "Historical Volatility":
    #         df = TechnicalIndicators.add_historical_volatility(df)
    #     elif feature == "Chandelier Exit":
    #         df = TechnicalIndicators.add_chandelier_exit(df)
    #     elif feature == "Keltner Channel":
    #         df = TechnicalIndicators.add_keltner_channel(df)
    #     elif feature == "Moving Average Envelope (MAE)":
    #         df = TechnicalIndicators.add_moving_average_envelope(df)
    #     elif feature == "Average Directional Index (ADX)":
    #         df = TechnicalIndicators.add_adx(df)
    #     elif feature == "Ichimoku Cloud":
    #         df = TechnicalIndicators.add_ichimoku_cloud(df)
    #     elif feature == "Parabolic SAR":
    #         df = TechnicalIndicators.add_parabolic_sar(df)
    #     elif feature == "Zigzag Indicator":
    #         df = TechnicalIndicators.add_zigzag_indicator(df)
    #     elif feature == "On-Balance Volume (OBV)":
    #         df = TechnicalIndicators.add_on_balance_volume(df)
    #     elif feature == "Volume Weighted Average Price (VWAP)":
    #         df = TechnicalIndicators.add_vwap(df)
    #     elif feature == "Accumulation/Distribution Line (ADL)":
    #         df = TechnicalIndicators.add_accumulation_distribution_line(df)
    #     elif feature == "Chaikin Money Flow (CMF)":
    #         df = TechnicalIndicators.add_chaikin_money_flow(df)
    #     elif feature == "Volume Oscillator":
    #         df = TechnicalIndicators.add_volume_oscillator(df)
    #     elif feature == "Awesome Oscillator":
    #         df = TechnicalIndicators.add_awesome_oscillator(df)
    #     elif feature == "TRIX":
    #         df = TechnicalIndicators.add_trix(df)
    #     elif feature == "Standard Pivot Points":
    #         df = TechnicalIndicators.add_standard_pivot_points(df)
        # ... [additional elif blocks for other indicators] ...
