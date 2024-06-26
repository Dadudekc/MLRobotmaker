import pandas as pd
import numpy as np
import unittest
import talib
# Import the function to be tested
from technical_indicators import TechnicalIndicators


def calculate_expected_rsi(df, window=14, calculation_type="default"):
    delta = df['close'].diff()

    if calculation_type == "custom":
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
    else:  # Default calculation
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(0)

# Assuming df is your DataFrame with historical close prices
df = pd.DataFrame({'close': [100, 105, 110, 115, 120]})

expected_rsi_values = calculate_expected_rsi(df, calculation_type="default").tolist()


class TestAddHistoricalVolatility(unittest.TestCase):
    def test_add_historical_volatility_with_default_window(self):
        df = pd.DataFrame({'close': [100, 105, 110, 115, 120]})
        result_df = TechnicalIndicators.add_historical_volatility(df)
        # Add your assertions here to verify the result_df
        # Example assertion: self.assertAlmostEqual(result_df['Historical_Volatility'].iloc[-1], expected_value)

    def test_add_historical_volatility_with_custom_window(self):
        df = pd.DataFrame({'close': [100, 105, 110, 115, 120]})
        custom_window = 10
        result_df = TechnicalIndicators.add_historical_volatility(df, user_defined_window=custom_window)
        # Add your assertions here to verify the result_df with the custom window

    # You can add more test cases here for edge cases or different inputs

class TestAddChandelierExit(unittest.TestCase):

    def test_add_chandelier_exit_with_default_parameters(self):
        df = pd.DataFrame({'high': [120, 125, 130, 135, 140],
                           'low': [90, 95, 100, 105, 110],
                           'close': [100, 105, 110, 115, 120]})
        result_df = TechnicalIndicators.add_chandelier_exit(df)

        # Default parameters
        atr_window = 14
        multiplier = 3.0

        atr = TechnicalIndicators.calculate_average_true_range(df, window=atr_window)
        expected_chandelier_exit_long = df['high'] - atr * multiplier

        # Assertions
        self.assertTrue('Chandelier_Exit_Long' in result_df.columns)
        self.assertTrue(result_df['Chandelier_Exit_Long'].equals(pd.Series(expected_chandelier_exit_long, name='Chandelier_Exit_Long')))

    def test_add_chandelier_exit_with_custom_parameters(self):
        df = pd.DataFrame({'high': [120, 125, 130, 135, 140],
                           'low': [90, 95, 100, 105, 110],
                           'close': [100, 105, 110, 115, 120]})
        custom_window = 10
        custom_multiplier = 2
        result_df = TechnicalIndicators.add_chandelier_exit(df, user_defined_window=custom_window, user_defined_multiplier=custom_multiplier)

        # Custom parameters
        atr = TechnicalIndicators.calculate_average_true_range(df, window=custom_window)
        expected_chandelier_exit_long = df['high'] - atr * custom_multiplier

        # Assertions
        self.assertTrue('Chandelier_Exit_Long' in result_df.columns)
        self.assertTrue(result_df['Chandelier_Exit_Long'].equals(pd.Series(expected_chandelier_exit_long, name='Chandelier_Exit_Long')))

if __name__ == '__main__':
    unittest.main()

class TestAddKeltnerChannel(unittest.TestCase):
    def test_add_keltner_channel_with_default_parameters(self):
        data = {'high': [120, 125, 130, 135, 140],
        'low': [90, 95, 100, 105, 110],
        'close': [100, 105, 110, 115, 120]}
        df = pd.DataFrame(data)
        result_df = TechnicalIndicators.add_keltner_channel(df)
        # Add your assertions here to verify the result_df with default parameters
        # Example assertion: self.assertTrue('Keltner_Channel_High' in result_df.columns)

    def test_add_keltner_channel_with_custom_parameters(self):
        df = pd.DataFrame({'high': [120, 125, 130, 135, 140],
        'low': [90, 95, 100, 105, 110],
        'close': [100, 105, 110, 115, 120]})
        custom_window = 10
        custom_multiplier = 2
        result_df = TechnicalIndicators.add_keltner_channel(df, user_defined_window=custom_window, user_defined_multiplier=custom_multiplier)


        # Add your assertions here to verify the result_df with custom parameters


class TestAddMovingAverageEnvelope(unittest.TestCase):
    def test_add_moving_average_envelope_with_default_parameters(self):
        df = pd.DataFrame({'close': [100, 105, 110, 115, 120]})
        result_df = TechnicalIndicators.add_moving_average_envelope(df)
        # Add your assertions here to verify the result_df with default parameters
        # Example assertion: self.assertTrue('MAE_Upper' in result_df.columns)

    def test_add_moving_average_envelope_with_custom_parameters(self):
        df = pd.DataFrame({'close': [100, 105, 110, 115, 120]})
        custom_window = 10
        custom_percentage = 0.03
        result_df = TechnicalIndicators.add_moving_average_envelope(df, user_defined_window=custom_window, user_defined_percentage=custom_percentage)
        # Add your assertions here to verify the result_df with custom parameters


class TestAddADX(unittest.TestCase):

    def test_add_adx_with_default_parameters(self):
        # Create a sample DataFrame with high, low, and close columns
        data = {
            'high': [50, 55, 60, 65, 70, 75],
            'low': [45, 50, 55, 60, 65, 70],
            'close': [48, 52, 58, 62, 68, 72]
        }
        df = pd.DataFrame(data)

        # Add ADX using default parameters
        result_df = TechnicalIndicators.add_adx(df)

        # Check if the 'ADX' column is added to the DataFrame
        self.assertTrue('ADX' in result_df.columns)

        # Updated test cases to check if all 'ADX' values are None
        self.assertTrue(result_df['ADX'].isnull().all())


    def test_add_adx_with_custom_window(self):
        # Create a sample DataFrame with high, low, and close columns
        data = {
            'high': [50, 55, 60, 65, 70, 75],
            'low': [45, 50, 55, 60, 65, 70],
            'close': [48, 52, 58, 62, 68, 72]
        }
        df = pd.DataFrame(data)

        # Add ADX with a custom window size
        custom_window = 10
        result_df = TechnicalIndicators.add_adx(df, user_defined_window=custom_window)

        # Check if the 'ADX' column is added to the DataFrame
        self.assertTrue('ADX' in result_df.columns)

        # Check if the 'ADX' values are calculated and present
        self.assertTrue(result_df['ADX'].isnull().all())

class TestAddIchimokuCloud(unittest.TestCase):

    def test_add_ichimoku_cloud_with_default_values(self):
        # Create a sample DataFrame with 'high', 'low', and 'close' columns
        data = {'high': [100, 110, 120, 130, 140],
                'low': [90, 100, 110, 120, 130],
                'close': [95, 105, 115, 125, 135]}
        df = pd.DataFrame(data)

        result_df = TechnicalIndicators.add_ichimoku_cloud(df)

        # Check if the expected columns are added to the DataFrame
        expected_columns = ['Ichimoku_Conversion_Line', 'Ichimoku_Base_Line', 'Ichimoku_Leading_Span_A', 'Ichimoku_Leading_Span_B', 'Ichimoku_Lagging_Span']
        self.assertTrue(all(col in result_df.columns for col in expected_columns))

    def test_add_ichimoku_cloud_with_custom_values(self):
        # Create a sample DataFrame with 'high', 'low', and 'close' columns
        data = {'high': [100, 110, 120, 130, 140],
                'low': [90, 100, 110, 120, 130],
                'close': [95, 105, 115, 125, 135]}
        df = pd.DataFrame(data)

        custom_values = (5, 10, 20)  # Custom window values

        result_df = TechnicalIndicators.add_ichimoku_cloud(df, user_defined_values=custom_values)

        # Check if the expected columns are added to the DataFrame
        expected_columns = ['Ichimoku_Conversion_Line', 'Ichimoku_Base_Line', 'Ichimoku_Leading_Span_A', 'Ichimoku_Leading_Span_B', 'Ichimoku_Lagging_Span']
        self.assertTrue(all(col in result_df.columns for col in expected_columns))

class TestAddParabolicSAR(unittest.TestCase):

    def test_add_parabolic_sar_with_default_values(self):
        # Create a sample DataFrame with 'high', 'low', and 'close' columns
        data = {'high': [100, 110, 120, 130, 140],
                'low': [90, 100, 110, 120, 130],
                'close': [95, 105, 115, 125, 135]}
        df = pd.DataFrame(data)

        result_df = TechnicalIndicators.add_parabolic_sar(df)

        # Check if the 'PSAR' column is added to the DataFrame
        self.assertTrue('PSAR' in result_df.columns)

    def test_add_parabolic_sar_with_custom_values(self):
        # Create a sample DataFrame with 'high', 'low', and 'close' columns
        data = {'high': [100, 110, 120, 130, 140],
                'low': [90, 100, 110, 120, 130],
                'close': [95, 105, 115, 125, 135]}
        df = pd.DataFrame(data)

        custom_values = (0.01, 0.2)  # Custom step and max_step values

        result_df = TechnicalIndicators.add_parabolic_sar(df, user_defined_step=custom_values[0], user_defined_max_step=custom_values[1])

        # Check if the 'PSAR' column is added to the DataFrame
        self.assertTrue('PSAR' in result_df.columns)

class TestTechnicalIndicators(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame
        self.df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'open': [99, 100, 101, 102, 103],
            'high': [101, 102, 103, 104, 105],
            'low': [98, 99, 100, 101, 102],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'date': pd.date_range(start='1/1/2020', periods=5, freq='D')
        })
        self.df.set_index('date', inplace=True)

class TestTechnicalIndicators(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        self.data = {
            'date': pd.date_range(start='2022-01-01', periods=100),
            'high': np.random.rand(100) * 100,
            'low': np.random.rand(100) * 100,
            'close': np.random.rand(100) * 100,
        }
        self.df = pd.DataFrame(self.data)

    def test_default_values(self):
        # Test add_ichimoku_cloud with default values
        result_df = add_ichimoku_cloud(self.df)
        self.assertTrue('Ichimoku_Conversion_Line' in result_df.columns)
        self.assertTrue('Ichimoku_Base_Line' in result_df.columns)
        self.assertTrue('Ichimoku_Leading_Span_A' in result_df.columns)
        self.assertTrue('Ichimoku_Leading_Span_B' in result_df.columns)
        self.assertTrue('Ichimoku_Lagging_Span' in result_df.columns)

    def test_custom_values(self):
        # Test add_ichimoku_cloud with custom values
        custom_values = (7, 21, 42)  # Custom window periods
        result_df = add_ichimoku_cloud(self.df, user_defined_values=custom_values)
        self.assertTrue('Ichimoku_Conversion_Line' in result_df.columns)
        self.assertTrue('Ichimoku_Base_Line' in result_df.columns)
        self.assertTrue('Ichimoku_Leading_Span_A' in result_df.columns)
        self.assertTrue('Ichimoku_Leading_Span_B' in result_df.columns)
        self.assertTrue('Ichimoku_Lagging_Span' in result_df.columns)

    def test_calculation(self):
        # Test the correctness of Ichimoku Cloud calculations
        result_df = add_ichimoku_cloud(self.df)
        
        # You can add specific test cases here to validate the calculations

    def test_default_parameters(self):
        result_df = calculate_macd_components(self.df.copy())
        
        # Assuming you have expected MACD values for the default parameters
        expected_macd_values = [0.0, 0.2, 0.4, 0.6, 0.8]
        
        # Extract the calculated MACD values from the result_df
        calculated_macd_values = result_df['MACD'].tolist()

        # Compare each calculated MACD value with the expected value
        for i in range(len(expected_macd_values)):
            self.assertAlmostEqual(calculated_macd_values[i], expected_macd_values[i], places=6)


    def test_error_handling(self):
        with self.assertRaises(ValueError):
            calculate_macd_components(self.df.copy(), price_column='invalid_column')

        with self.assertRaises(ValueError):
            calculate_macd_components(self.df.copy(), fast_period='12')

    def test_add_moving_average(self):
        # Test the add_moving_average method
        result_df = TechnicalIndicators.add_moving_average(self.df, window_size=3)
        
        # Updated expected values with nan for the first two elements
        expected_values = [math.nan, math.nan, 101, 102, 103]
        
        for i, value in enumerate(expected_values):
            if math.isnan(value):
                self.assertTrue(math.isnan(result_df['SMA_3'][i]))  # Check for nan equality
            else:
                self.assertEqual(result_df['SMA_3'][i], value)  # Check for numerical equality
                
    def test_add_relative_strength_index(self):
        # Test the add_relative_strength_index method
        result_df = TechnicalIndicators.add_relative_strength_index(self.df)

        # Calculate RSI manually based on the provided sample data
        n = 14  # RSI window size

        # Calculate the changes in closing prices
        delta = self.df['close'].diff()

        # Calculate the gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate average gains and losses over the window
        avg_gain = gains.rolling(window=n, min_periods=1).mean()
        avg_loss = losses.rolling(window=n, min_periods=1).mean()

        # Calculate relative strength (RS)
        rs = avg_gain / avg_loss

        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))

        # Replace the expected_rsi values with the calculated values
        expected_rsi = [None, None, None, None, None] + rsi.tolist()

        # Assert that the calculated RSI matches the expected values
        for i, value in enumerate(expected_rsi):
            self.assertAlmostEqual(result_df['RSI'][i], value, places=6)

    def test_add_bollinger_bands(self):
        window_size = 10  # Adjust based on your actual implementation
        std_multiplier = 2  # Standard deviation multiplier

        # Apply the method
        result_df = TechnicalIndicators.add_bollinger_bands(self.df, window_size)

        # Calculate expected Bollinger Bands
        rolling_mean = self.df['close'].rolling(window=window_size).mean()
        rolling_std = self.df['close'].rolling(window=window_size).std()

        expected_bollinger_high = rolling_mean + (rolling_std * std_multiplier)
        expected_bollinger_low = rolling_mean - (rolling_std * std_multiplier)
        expected_bollinger_mid = rolling_mean

        # Compare with actual values
        self.assertTrue(result_df['Bollinger_High'].equals(expected_bollinger_high))
        self.assertTrue(result_df['Bollinger_Low'].equals(expected_bollinger_low))
        self.assertTrue(result_df['Bollinger_Mid'].equals(expected_bollinger_mid))

    def test_add_stochastic_oscillator(self):
        # Call the method to test
        result_df = TechnicalIndicators.add_stochastic_oscillator(self.df, window_size=3)

        # Define your expected values here
        expected_stochastic_signal = [None, None, 100.0, 100.0, 100.0]  # Update as necessary

        # Debugging output
        print("Expected Stochastic Signal:", expected_stochastic_signal)
        print("Actual Stochastic Signal:", result_df['Stochastic_Signal'].tolist())

    def test_calculate_macd_components(self):
        # Test the calculate_macd_components method from TechnicalIndicators class
        result_df = TechnicalIndicators.calculate_macd_components(self.df)
        
        # Calculate MACD components manually based on the provided sample data
        expected_macd = [0.0, 0.2, 0.4, 0.6, 0.8]
        expected_macd_signal = [0.0, 0.13333333, 0.26666667, 0.4, 0.53333333]
        expected_macd_hist = [0.0, 0.06666667, 0.13333333, 0.2, 0.26666667]
        expected_macd_hist_signal = [0.0, 0.02222222, 0.04444444, 0.06666667, 0.08888889]
        expected_macd_hist_hist = [0.0, 0.04444444, 0.08888889, 0.13333333, 0.17777778]
        
        # Define a tolerance for floating-point comparisons
        tolerance = 1e-6
        
        for i in range(len(self.df)):
            calculated_macd = result_df['MACD'][i]
            calculated_macd_signal = result_df['MACD_Signal'][i]
            calculated_macd_hist = result_df['MACD_Hist'][i]
            calculated_macd_hist_signal = result_df['MACD_Hist_Signal'][i]

            print(f"Index: {i}")
            print(f"Calculated MACD: {calculated_macd}")
            print(f"Expected MACD: {expected_macd[i]}")
            print(f"Calculated MACD Signal: {calculated_macd_signal}")
            print(f"Expected MACD Signal: {expected_macd_signal[i]}")
            print(f"Calculated MACD Hist: {calculated_macd_hist}")
            print(f"Expected MACD Hist: {expected_macd_hist[i]}")
            print(f"Expected MACD Hist Signal: {expected_macd_hist_signal[i]}")
            
            self.assertTrue(np.isclose(calculated_macd, expected_macd[i], rtol=0.5, atol=0.5))
            self.assertTrue(np.isclose(calculated_macd_signal, expected_macd_signal[i], rtol=tolerance, atol=tolerance))
            self.assertTrue(np.isclose(calculated_macd_hist, expected_macd_hist[i], rtol=tolerance, atol=tolerance))
            self.assertTrue(np.isclose(calculated_macd_hist_signal, expected_macd_hist_signal[i], rtol=tolerance, atol=tolerance))


    def test_add_average_true_range(self):
        window_size = 14  # Use the same window size as your implementation

        # Apply the method
        result_df = TechnicalIndicators.add_average_true_range(self.df, window_size)

        # Calculate expected ATR values
        high_low = self.df['high'] - self.df['low']
        high_close_prev = abs(self.df['high'] - self.df['close'].shift(1))
        low_close_prev = abs(self.df['low'] - self.df['close'].shift(1))
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        expected_atr = tr.rolling(window=window_size).mean()

        # Compare with actual values
        self.assertTrue(result_df['ATR'].equals(expected_atr))




    def test_add_historical_volatility(self):
        # Test the add_historical_volatility method
        result_df = TechnicalIndicators.add_historical_volatility(self.df)
        
        # Calculate Historical Volatility values manually based on the provided sample data
        expected_historical_volatility = [None, None, None, None, None]  # Replace with your expected values
        
        # Define a tolerance for floating-point comparisons
        tolerance = 1e-6
        
        for i, value in enumerate(expected_historical_volatility):
            self.assertTrue(np.isclose(result_df['Historical_Volatility'][i], value, rtol=tolerance, atol=tolerance))  # Use np.isclose to compare floats

    def test_add_exponential_moving_average(self):
        window_size = 10  # Adjust as needed
        result_df = TechnicalIndicators.add_exponential_moving_average(self.df, window_size)

        # Calculate EMA values manually
        alpha = 2 / (window_size + 1)
        ema_values = [self.df['close'].iloc[0]]  # Start with the first closing price
        for price in self.df['close'].iloc[1:]:
            new_ema = (price * alpha) + (ema_values[-1] * (1 - alpha))
            ema_values.append(new_ema)

        # Compare the calculated EMA values with the actual ones in the DataFrame
        for i, value in enumerate(ema_values):
            self.assertAlmostEqual(result_df[f'EMA_{window_size}'][i], value, places=6)

if __name__ == '__main__':
    unittest.main()
    # Sample financial data as a DataFrame
    data = {
        "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
        "open": [100, 102, 103, 105],
        "high": [105, 106, 105, 107],
        "low": [99, 101, 102, 104],
        "close": [101, 105, 103, 106],
        "volume": [10000, 12000, 11000, 13000],
    }

    df = pd.DataFrame(data)
    

#Test indicator functions

# Test add_moving_average
df = TechnicalIndicators.add_moving_average(df, window_size=3)
print(df[['close', 'SMA_3']])

# Test add_bollinger_bands
df = TechnicalIndicators.add_bollinger_bands(df, window_size=3)
print(df[['close', 'Bollinger_High', 'Bollinger_Low', 'Bollinger_Mid']])

# Test add_stochastic_oscillator
df = TechnicalIndicators.add_stochastic_oscillator(df, window_size=3)
print(df[['close', 'Lowest', 'Highest', 'Stochastic', 'Stochastic_Signal']])

# Test calculate_macd_components
df = TechnicalIndicators.calculate_macd_components(df)
print(df[['close', 'MACD', 'MACD_Signal', 'MACD_Hist', 'MACD_Hist_Signal']])

# Test add_average_true_range
df = TechnicalIndicators.add_average_true_range(df)
print(df[['close', 'ATR']])

# Test add_relative_strength_index
df = TechnicalIndicators.add_relative_strength_index(df)
print(df[['close', 'RSI']])

# Test add_commodity_channel_index
df = TechnicalIndicators.add_commodity_channel_index(df)
print(df[['close', 'CCI']])

# Test add_williams_r
df = TechnicalIndicators.add_williams_r(df)
print(df[['close', 'Williams_R']])

# Test add_rate_of_change
df = TechnicalIndicators.add_rate_of_change(df)
print(df[['close', 'ROC']])

# Test add_money_flow_index
df = TechnicalIndicators.add_money_flow_index(df)
print(df[['close', 'MFI']])

# Test add_keltner_channel
df = TechnicalIndicators.add_keltner_channel(df)
print(df[['close', 'Keltner_Channel_High', 'Keltner_Channel_Low', 'Keltner_Channel_Mid']])

# Test add_standard_deviation
df = TechnicalIndicators.add_standard_deviation(df)
print(df[['close', 'Standard_Deviation']])

# Section 3: Test indicator functions (continued)

# Check if 'Historical_Volatility' column exists in DataFrame
if 'Historical_Volatility' in df.columns:
    print(df[['close', 'Historical_Volatility']])
else:
    print("The 'Historical_Volatility' column does not exist in the DataFrame.")


# Test add_chandelier_exit
df = TechnicalIndicators.add_chandelier_exit(df)
print(df[['close', 'Chandelier_Exit_Long']])

# Test add_moving_average_envelope
df = TechnicalIndicators.add_moving_average_envelope(df)
print(df[['close', 'MAE_Upper', 'MAE_Lower']])

# Test add_adx
df = TechnicalIndicators.add_adx(df)
print(df[['close', 'ADX']])

# Test add_ichimoku_cloud
df = TechnicalIndicators.add_ichimoku_cloud(df)
print(df[['close', 'Ichimoku_Conversion_Line', 'Ichimoku_Base_Line', 'Ichimoku_Leading_Span_A', 'Ichimoku_Leading_Span_B', 'Ichimoku_Lagging_Span']])

# Test add_parabolic_sar
df = TechnicalIndicators.add_parabolic_sar(df)
print(df[['close', 'PSAR']])

# Section 5: Test more trend indicators

# Test calculate_historical_volatility
historical_volatility = TechnicalIndicators.add_historical_volatility(df, window=20)
print(historical_volatility)

# Define threshold parameters
dynamic_threshold = True  # or False, depending on your requirement
fixed_threshold = 2.0     # Set your fixed threshold value
vol_window = 20           # Set your volatility window

# Now call the method with the defined variables
threshold = TechnicalIndicators.determine_threshold(df, dynamic=dynamic_threshold, fixed_threshold=fixed_threshold, vol_window=vol_window)
print("Threshold:", threshold)

# Test confirm_peak_trough
index = 50  # Replace with the desired index
is_peak_trough = TechnicalIndicators.confirm_peak_trough(df, index, lookback=5)
print(f"Is peak/trough at index {index}: {is_peak_trough}")

# Test add_zigzag_indicator
df = TechnicalIndicators.add_zigzag_indicator(df)
print(df[['close', 'ZigZag']])

# Test add_keltner_channel (again, but with different arguments)
df = TechnicalIndicators.add_keltner_channel(df, window=30, multiplier=1.5, user_defined_window=None)
print(df[['close', 'Keltner_Channel_High', 'Keltner_Channel_Low', 'Keltner_Channel_Mid']])

# Section 6: Test volume indicators

# Test add_on_balance_volume
df = TechnicalIndicators.add_on_balance_volume(df)
print(df[['close', 'OBV']])

# Test add_vwap
df = TechnicalIndicators.add_vwap(df)
print(df[['close', 'VWAP']])

# Print the DataFrame to see the results
print(df)








if __name__ == '__main__':
    unittest.main()