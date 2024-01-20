#visualization.py

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

def create_candlestick_chart(df: pd.DataFrame, title: str = "Candlestick Chart", save_path: str = None):
    """
    Create a candlestick chart from the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing OHLCV data.
        title (str, optional): The title of the chart. Defaults to "Candlestick Chart".
        save_path (str, optional): The path to save the chart as an image file. Defaults to None.

    Raises:
        ValueError: If the required OHLCV columns are not found in the DataFrame.
    """
    try:
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        if not required_columns.issubset(df.columns):
            raise ValueError("DataFrame must contain OHLCV columns.")
        
        mpf.plot(df, type='candle', title=title, savefig=save_path)
    except Exception as e:
        print(f"Error in creating candlestick chart: {e}")

def create_line_chart(df: pd.DataFrame, x_column: str = 'date', y_column: str = 'close', title: str = "Line Chart", color: str = 'blue'):
    """
    Create a line chart from the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing data for the line chart.
        x_column (str, optional): The column to use for the x-axis. Defaults to 'date'.
        y_column (str, optional): The column to use for the y-axis. Defaults to 'close'.
        title (str, optional): The title of the chart. Defaults to "Line Chart".
        color (str, optional): The color of the line. Defaults to 'blue'.

    Returns:
        None

    Raises:
        ValueError: If the specified x_column or y_column is not found in the DataFrame.
    """
    try:
        if x_column not in df.columns:
            raise ValueError(f"Column '{x_column}' not found in DataFrame.")
        if y_column not in df.columns:
            raise ValueError(f"Column '{y_column}' not found in DataFrame.")
        
        plt.figure(figsize=(12, 6))
        plt.plot(df[x_column], df[y_column], label=f'{y_column} Price', color=color)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Error in creating line chart: {e}")
