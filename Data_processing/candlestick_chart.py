import mplfinance as mpf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import Toplevel
import pandas as pd

class CandlestickChart:
    def __init__(self, parent, data=None, style='yahoo', mav=(3, 6, 9)):
        self.parent = parent
        self.new_window = Toplevel(parent)  # Create a new window
        self.new_window.title("Candlestick Chart")
        self.style = style
        self.mav = mav
        self.fig, self.axes = None, None
        self.canvas = None
        self.crosshair_enabled = False
        self.crosshair_line_x = None
        self.crosshair_line_y = None
        self.data = self.prepare_data(data) if data is not None else None

    def prepare_data(self, data):
        # Rename columns if needed
        columns_map = {'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close'}
        data.rename(columns=columns_map, inplace=True)

        # Convert 'date' column to datetime and set it as the index
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)

        # Sort the DataFrame by date in ascending order
        data.sort_index(inplace=True)

        return self.validate_data(data)    
        
    def create_chart(self):
        if self.data is not None:
            plot_kwargs = {
                'type': 'candle',
                'mav': self.mav,
                'style': self.style,
                'returnfig': True
            }
            if 'volume' in self.data.columns:
                plot_kwargs['volume'] = True
            else:
                print("Volume data not found, creating chart without volume.")

            self.fig, self.axes = mpf.plot(self.data, **plot_kwargs)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.new_window)  # Use new_window as master
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill="both", expand=True)


    def update_data(self, new_data):
        self.data = self.prepare_data(new_data) if new_data is not None else None
        self.update_chart()

    def update_chart(self):
        if self.data is not None:
            self.axes.clear()
            mpf.plot(self.data, ax=self.axes, type='candle', mav=self.mav, volume=True, style=self.style)
            self.canvas.draw()

    def validate_data(self, data):
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in the data: {missing_columns}")
        return data

    def set_style(self, new_style):
        self.style = new_style
        self.update_chart()

    def set_moving_averages(self, new_mav):
        self.mav = new_mav
        self.update_chart()

    def toggle_moving_averages(self, show_mav=True):
        self.show_mav = show_mav
        self.update_chart()

    def zoom_in(self):
        # Increase the visible data range and update the chart
        self.visible_data_range += 10  # You can adjust the zoom level
        self.update_chart()

    def zoom_out(self):
        # Decrease the visible data range and update the chart
        self.visible_data_range -= 10  # You can adjust the zoom level
        self.update_chart()

    def set_color_scheme(self, new_color_scheme):
        self.color_scheme = new_color_scheme
        self.update_chart()

    def toggle_volume_bars(self, show_volume=True):
        self.show_volume = show_volume
        self.update_chart()

    def set_date_range(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.update_chart()

    def enable_crosshair(self):
        if not self.crosshair_enabled:
            self.crosshair_enabled = True
            self.parent.bind("<Motion>", self.track_crosshair)
            self.draw_crosshair()

    def disable_crosshair(self):
        if self.crosshair_enabled:
            self.crosshair_enabled = False
            self.parent.unbind("<Motion>")
            self.clear_crosshair()

    def track_crosshair(self, event):
        if self.crosshair_enabled:
            x, y = event.x, event.y
            self.crosshair_line_x.set_xdata(x)
            self.crosshair_line_y.set_ydata(y)
            self.canvas.draw()

    def draw_crosshair(self):
        # Initialize horizontal and vertical crosshair lines
        self.crosshair_line_x = self.axes.axhline(0, color='gray', linestyle='--')
        self.crosshair_line_y = self.axes.axvline(0, color='gray', linestyle='--')
        self.canvas.draw()

    def clear_crosshair(self):
        if self.crosshair_line_x:
            self.crosshair_line_x.remove()
            self.crosshair_line_x = None
        if self.crosshair_line_y:
            self.crosshair_line_y.remove()
            self.crosshair_line_y = None
        self.canvas.draw()

    def draw_trendline(self, start_point, end_point, color='blue'):
        if self.data is not None:
            x1, y1 = start_point
            x2, y2 = end_point
            trendline_data = pd.DataFrame({'Date': [self.data.index[0], self.data.index[-1]],
                                          'Trendline': [y1, y2]})
            trendline_data.set_index('Date', inplace=True)
            self.axes.plot(trendline_data.index, trendline_data['Trendline'], color=color, label='Trendline')
            self.canvas.draw()

    def draw_annotation(self, x, y, text):
        if self.data is not None:
            self.axes.annotate(text, xy=(x, y), xytext=(x, y + 2), textcoords='data', fontsize=10,
                               arrowprops=dict(facecolor='black', shrink=0.05),
                               bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='lightyellow'))
            self.canvas.draw()

# Add any additional methods or functionality as needed.
