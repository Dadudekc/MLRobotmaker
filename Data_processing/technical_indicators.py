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
import self

class AverageTrueRange:
    def __init__(self, high, low, close, window_size):
        self.high = high
        self.low = low
        self.close = close
        self.window_size = window_size

    def true_range(self):
        high_low = self.high - self.low
        high_close_prev = abs(self.high - self.close.shift(1))
        low_close_prev = abs(self.low - self.close.shift(1))
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1)
        return tr.max(axis=1)

    def average_true_range(self):
        tr = self.true_range()
        atr = tr.rolling(window=self.window_size).mean()
        return atr

class TechnicalIndicators:
    # Section 1: Basic indicators
    @staticmethod
    def add_moving_average(df, window_size=10, user_defined_window=None, column='close'):
        # Override window size if user defined window is provided
        if user_defined_window is not None:
            window_size = user_defined_window
        
        # Check if the specified column exists
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        # Calculate the moving average and add it as a new column
        df[f'SMA_{window_size}'] = df[column].rolling(window=window_size).mean()
        return df


    @staticmethod
    def add_bollinger_bands(df, window_size=10, std_multiplier=2, user_defined_window=None):
        if user_defined_window is not None:
            window_size = user_defined_window
        rolling_mean = df['close'].rolling(window=window_size).mean()
        rolling_std = df['close'].rolling(window=window_size).std()
        df['Bollinger_High'] = rolling_mean + (rolling_std * std_multiplier)
        df['Bollinger_Low'] = rolling_mean - (rolling_std * std_multiplier)
        df['Bollinger_Mid'] = rolling_mean
        return df

    @staticmethod
    def add_exponential_moving_average(df, column='close', window_size=10):
        ema = df[column].ewm(span=window_size, adjust=False).mean()
        df['EMA_' + str(window_size)] = ema
        return df

    @staticmethod
    def add_stochastic_oscillator(df, window_size=14, user_defined_window=None):
        if user_defined_window is not None:
            window_size = user_defined_window
        df['Lowest'] = df['low'].rolling(window=window_size).min()
        df['Highest'] = df['high'].rolling(window=window_size).max()
        df['Stochastic'] = 100 * ((df['close'] - df['Lowest']) / (df['Highest'] - df['Lowest']))
        df['Stochastic_Signal'] = df['Stochastic'].rolling(window=3).mean()
        return df

    @staticmethod
    def calculate_macd_components(df, fast_period=12, slow_period=26, signal_period=9, price_column='close'):
        """
        Calculate MACD components for a given DataFrame.

        Args:
            df (DataFrame): Stock price data.
            fast_period (int): The fast period for EMA calculation.
            slow_period (int): The slow period for EMA calculation.
            signal_period (int): The signal period for EMA calculation.
            price_column (str): Column name for price data in df.

        Returns:
            DataFrame: DataFrame with MACD components added.
        """

        # Validate input parameters
        if not all(isinstance(x, int) for x in [fast_period, slow_period, signal_period]):
            raise ValueError("Period parameters must be integers.")
        if price_column not in df.columns:
            raise ValueError(f"{price_column} column not found in DataFrame.")

        # Calculate MACD components
        fast_ema = df[price_column].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df[price_column].ewm(span=slow_period, adjust=False).mean()
        df['MACD'] = fast_ema - slow_ema
        df['MACD_Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        df['MACD_Hist_Signal'] = df['MACD_Hist'].ewm(span=signal_period, adjust=False).mean()

        return df

    @staticmethod
    def add_average_true_range(df, window_size=14, user_defined_window=None):
        if user_defined_window is not None:
            window_size = user_defined_window

        try:
            indicator = AverageTrueRange(df['high'], df['low'], df['close'], window_size)
            df['ATR'] = indicator.average_true_range()
            return df
        except NameError:
            raise NameError("AverageTrueRange class is not defined. Make sure to define the class or import it from the appropriate module.")

    # Section 2: Other Oscillator Indicators
    @staticmethod
    def add_relative_strength_index(df, window=14, user_defined_window=None, calculation_type="default"):
        if user_defined_window is not None:
            window = user_defined_window

        delta = df['close'].diff()

        if calculation_type == "custom":
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        else:  # Default calculation
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

        df['RSI'] = rsi.fillna(0)
        return df

    @staticmethod
    def add_commodity_channel_index(df, window=20, user_defined_window=None):
        if user_defined_window is not None:
            window = user_defined_window
        tp = (df['high'] + df['low'] + df['close']) / 3
        cci = (tp - tp.rolling(window=window).mean()) / (0.015 * tp.rolling(window=window).std())
        df['CCI'] = cci.fillna(0)
        return df

    @staticmethod
    def add_williams_r(df, window=14, user_defined_window=None):
        if user_defined_window is not None:
            window = user_defined_window
        highest_high = df['high'].rolling(window=window).max()
        lowest_low = df['low'].rolling(window=window).min()
        df['Williams_R'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
        return df

    @staticmethod
    def add_rate_of_change(df, window=10, user_defined_window=None):
        if user_defined_window is not None:
            window = user_defined_window
        df['ROC'] = ((df['close'] - df['close'].shift(window)) / df['close'].shift(window)) * 100
        return df

    @staticmethod
    def add_money_flow_index(df, window=14, user_defined_window=None):
        if user_defined_window is not None:
            window = user_defined_window
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(df['close'] > df['close'].shift(1), 0).rolling(window=window).sum()
        negative_flow = money_flow.where(df['close'] < df['close'].shift(1), 0).rolling(window=window).sum()
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        df['MFI'] = mfi.fillna(0)
        return df

    # Section 3: Volatility Indicators
    @staticmethod
    def add_standard_pivot_points(df, high='high', low='low', close='close'):
        """
        Add Standard Pivot Points to the DataFrame.

        :param df: Pandas DataFrame with 'high', 'low', and 'close' columns.
        :param high: Column name for high prices.
        :param low: Column name for low prices.
        :param close: Column name for close prices.
        :return: DataFrame with Pivot Points added.
        """
        if not all(column in df.columns for column in [high, low, close]):
            raise ValueError("DataFrame must contain high, low, and close columns")

        # Calculate Pivot Points
        df['Pivot_Point'] = (df[high].shift(1) + df[low].shift(1) + df[close].shift(1)) / 3
        df['R1'] = 2 * df['Pivot_Point'] - df[low].shift(1)  # First Resistance Level
        df['S1'] = 2 * df['Pivot_Point'] - df[high].shift(1)  # First Support Level
        df['R2'] = df['Pivot_Point'] + (df[high].shift(1) - df[low].shift(1))  # Second Resistance Level
        df['S2'] = df['Pivot_Point'] - (df[high].shift(1) - df[low].shift(1))  # Second Support Level
        df['R3'] = df['high'].shift(1) + 2 * (df['Pivot_Point'] - df['low'].shift(1))  # Third Resistance Level
        df['S3'] = df['low'].shift(1) - 2 * (df['high'].shift(1) - df['Pivot_Point'])  # Third Support Level

        return df

    @staticmethod
    def add_standard_deviation(df, window_size=20, user_defined_window=None):
        if user_defined_window is not None:
            window_size = user_defined_window
        df['Standard_Deviation'] = df['close'].rolling(window=window_size).std()
        return df

    @staticmethod
    def add_historical_volatility(df, window=20, user_defined_window=None):
        """
        Adds historical volatility to the DataFrame.

        Args:
        df (pd.DataFrame): DataFrame containing the 'close' prices.
        window (int, optional): The window size for calculating volatility. Defaults to 20.
        user_defined_window (int, optional): User defined window size. If provided, it overrides the default.

        Returns:
        pd.DataFrame: DataFrame with the new 'Historical_Volatility' column.
        """

        # Validate if 'close' column exists in the DataFrame
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column")

        # Override window size if user_defined_window is provided
        if user_defined_window is not None:
            window = user_defined_window

        # Calculate log returns
        log_return = np.log(df['close'] / df['close'].shift(1))

        # Calculate and add historical volatility column
        df['Historical_Volatility'] = log_return.rolling(window=window).std() * np.sqrt(window)

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
        if user_defined_window is not None:
            window = user_defined_window
        if user_defined_multiplier is not None:
            multiplier = user_defined_multiplier

        # Check if the required columns are present in the DataFrame
        required_columns = ['high', 'low', 'close']
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {', '.join(required_columns)}")

        # Calculate the Keltner Channel
        ma = df['close'].rolling(window=window).mean()
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=window)
        df['Keltner_Channel_High'] = ma + (multiplier * atr)
        df['Keltner_Channel_Low'] = ma - (multiplier * atr)
        df['Keltner_Channel_Mid'] = ma

        return df


    @staticmethod
    # Section 4: Trend Indicators
    def add_moving_average_envelope(df, window_size=10, percentage=0.025, user_defined_window=None, user_defined_percentage=None):
        # Check if the user wants to override the defaults
        if user_defined_window is not None:
            window_size = user_defined_window
        if user_defined_percentage is not None:
            percentage = user_defined_percentage
            
        SMA = df['close'].rolling(window=window_size).mean()
        df['MAE_Upper'] = SMA + (SMA * percentage)
        df['MAE_Lower'] = SMA - (SMA * percentage)
        return df

    @staticmethod   
    def add_adx(df, window=14, user_defined_window=None):
        if user_defined_window is not None:
            window = user_defined_window

        if len(df) >= window:
            adx = ADXIndicator(df['high'], df['low'], df['close'], window)
            df['ADX'] = adx.adx()
        else:
            df['ADX'] = None  # Fill with a default value (you can use None or another appropriate default)

        return df


    @staticmethod   
    def add_ichimoku_cloud(df, user_defined_values=None):
        if user_defined_values is not None:
            nine_window, twenty_six_window, fifty_two_window = user_defined_values
        else:
            nine_window, twenty_six_window, fifty_two_window = 9, 26, 52

        nine_period_high = df['high'].rolling(window=nine_window).max()
        nine_period_low = df['low'].rolling(window=nine_window).min()
        df['Ichimoku_Conversion_Line'] = (nine_period_high + nine_period_low) / 2

        twenty_six_period_high = df['high'].rolling(window=twenty_six_window).max()
        twenty_six_period_low = df['low'].rolling(window=twenty_six_window).min()
        df['Ichimoku_Base_Line'] = (twenty_six_period_high + twenty_six_period_low) / 2

        df['Ichimoku_Leading_Span_A'] = ((df['Ichimoku_Conversion_Line'] + df['Ichimoku_Base_Line']) / 2).shift(twenty_six_window)

        fifty_two_period_high = df['high'].rolling(window=fifty_two_window).max()
        fifty_two_period_low = df['low'].rolling(window=fifty_two_window).min()
        df['Ichimoku_Leading_Span_B'] = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(twenty_six_window)

        df['Ichimoku_Lagging_Span'] = df['close'].shift(-twenty_six_window)

        return df

    @staticmethod
    def add_parabolic_sar(df, step=0.02, max_step=0.2):
        # Ensure data consistency
        if df.isnull().values.any():
            raise ValueError("DataFrame contains NaN values.")

        # Check DataFrame structure
        print("DataFrame columns:", df.columns)
        print("Sample data:", df.head())

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
            # Debugging information
            print(f"Processing index {i}: PSAR={psar}, High={psar_high}, Low={psar_low}")

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
        if dynamic:
            TechnicalIndicators.add_historical_volatility(df, window=vol_window)

            # Debugging: Print the type and head of the Historical_Volatility column
            print("Historical_Volatility column type:", df['Historical_Volatility'].dtype)
            print("Historical_Volatility column sample:", df['Historical_Volatility'].head())

            volatility = df['Historical_Volatility'].fillna(0)
            
            # Debugging: Print the type and head of the volatility series
            print("Volatility series type:", type(volatility))
            print("Volatility series sample:", volatility.head())

            threshold = volatility * fixed_threshold

            # Debugging: Print the type and head of the threshold series
            print("Threshold series type:", type(threshold))
            print("Threshold series sample:", threshold.head())

            return threshold
        else:
            fixed_threshold_series = pd.Series([fixed_threshold] * len(df), index=df.index)
            # Debugging: Print the type and head of the fixed threshold series
            print("Fixed threshold series type:", type(fixed_threshold_series))
            print("Fixed threshold series sample:", fixed_threshold_series.head())
            return fixed_threshold_series



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
            current_threshold = threshold.iloc[i]['Historical_Volatility']

            # Debugging: Print the types and values of variables used in the comparison
            print(f"Index: {i}, Current Close: {current_close}, Previous Close: {previous_close}, Current Threshold: {current_threshold}")

            if pd.notna(current_threshold) and abs(current_close - previous_close) > current_threshold:
                df.loc[i, 'ZigZag'] = current_close

        return df


# Section 6: Volume Indicators
    @staticmethod
    def add_on_balance_volume(df, user_defined_window=None):
        if user_defined_window is not None:
            window = user_defined_window
        else:
            window = 14  # Default value
        
        df['OBV'] = (df['volume'] * (~df['close'].diff().le(0) * 2 - 1)).cumsum()
        return df
    @staticmethod  
    def add_vwap(df, user_defined_window=None):
        if user_defined_window is not None:
            window = user_defined_window
        else:
            window = 20  # Default value
        
        df['Cumulative_Volume_Price'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum()
        df['Cumulative_Volume'] = df['volume'].cumsum()
        df['VWAP'] = df['Cumulative_Volume_Price'] / df['Cumulative_Volume']
        return df
    @staticmethod
    def add_accumulation_distribution_line(df, user_defined_window=None):
        if user_defined_window is not None:
            window = user_defined_window
        else:
            window = 20  # Default value
        
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv.fillna(0, inplace=True)
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
    @staticmethod
    def add_volume_oscillator(df, short_window=None, long_window=None):
        if short_window is None:
            short_window = 12  # Default value
        if long_window is None:
            long_window = 26  # Default value
        
        short_vol = df['volume'].rolling(window=short_window).mean()
        long_vol = df['volume'].rolling(window=long_window).mean()
        df['Volume_Oscillator'] = short_vol - long_vol
        return df

    # Add more volume indicators as needed...

# Section 7: Other Indicators
    @staticmethod
    def add_awesome_oscillator(df, short_window=5, long_window=34):
        if short_window is None:
            short_window = 5  # Default value
        if long_window is None:
            long_window = 34  # Default value
        
        mid_price = (df['high'] + df['low']) / 2
        df['Awesome_Oscillator'] = mid_price.rolling(window=short_window).mean() - mid_price.rolling(window=long_window).mean()
        return df

    @staticmethod
    def add_trix(df, span=15, signal_line_span=9):
        """
        Add TRIX and its signal line to the DataFrame.

        :param df: Pandas DataFrame with a 'close' column.
        :param span: The span for calculating TRIX. Default is 15.
        :param signal_line_span: The span for calculating the signal line. Default is 9.
        :return: DataFrame with TRIX and signal line columns added.
        """
        # Calculate the first EMA
        ema1 = df['close'].ewm(span=span, adjust=False).mean()

        # Calculate the second EMA
        ema2 = ema1.ewm(span=span, adjust=False).mean()

        # Calculate the third EMA
        ema3 = ema2.ewm(span=span, adjust=False).mean()

        # Calculate TRIX
        df['trix'] = 100 * (ema3.pct_change())

        # Calculate the signal line (EMA of TRIX)
        df['trix_signal'] = df['trix'].ewm(span=signal_line_span, adjust=False).mean()

        return df



# Section 8: Custom Indicators
    @staticmethod
    def add_custom_indicator(df, indicator_name, indicator_function, *args, **kwargs):
        df = indicator_function(df, *args, **kwargs)
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

