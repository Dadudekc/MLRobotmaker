#visualization.py

import matplotlib.pyplot as plt
import mplfinance as mpf

def create_candlestick_chart(df, title="Candlestick Chart"):
    """
    Create a candlestick chart from the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing OHLCV data.
        title (str): The title of the chart.

    Returns:
        None
    """
    mpf.plot(df, type='candle', title=title)

def create_line_chart(df, title="Line Chart"):
    """
    Create a line chart from the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing data for the line chart.
        title (str): The title of the chart.

    Returns:
        None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['close'], label='Close Price', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
