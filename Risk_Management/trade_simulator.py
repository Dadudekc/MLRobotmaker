# trade_simulator.py

import pandas as pd

class TradeSimulator:
    def __init__(self, initial_capital=100000, commission=0.001, slippage=0.001, leverage=1):
        self.trades = []
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.leverage = leverage
        self.cash = initial_capital
        self.positions = {}

    def simulate_trade(self, asset, amount, trade_type, price, risk_factor, trade_option_type=None):
        try:
            amount = float(amount)
            if trade_type not in ['buy', 'sell']:
                raise ValueError("Invalid trade type. Use 'buy' or 'sell'.")

            trade_value = amount * price * self.leverage
            transaction_cost = trade_value * (self.commission + self.slippage)
            net_trade_value = trade_value - transaction_cost if trade_type == 'buy' else trade_value + transaction_cost

            if trade_type == 'buy':
                if self.cash < net_trade_value:
                    raise ValueError("Not enough cash to complete the trade.")
                self.cash -= net_trade_value
                if asset in self.positions:
                    self.positions[asset] += amount
                else:
                    self.positions[asset] = amount
            elif trade_type == 'sell':
                if asset not in self.positions or self.positions[asset] < amount:
                    raise ValueError("Not enough positions to sell.")
                self.cash += net_trade_value
                self.positions[asset] -= amount
                if self.positions[asset] == 0:
                    del self.positions[asset]

            trade = {
                'asset': asset,
                'amount': amount,
                'price': price,
                'type': trade_type,
                'risk_factor': risk_factor,
                'option_type': trade_option_type,
                'value': net_trade_value if trade_type == 'buy' else -net_trade_value
            }
            self.trades.append(trade)
        except ValueError as e:
            print(f"Error simulating trade: {e}")

    def calculate_portfolio_value(self):
        portfolio_value = self.cash
        for asset, amount in self.positions.items():
            latest_price = self.get_latest_price(asset)
            portfolio_value += amount * latest_price
        return portfolio_value

    def get_latest_price(self, asset):
        if asset in self.historical_data:
            return self.historical_data[asset]['Close'].iloc[-1]
        return 0

    def generate_trade_report(self):
        report = pd.DataFrame(self.trades)
        return report

    def backtest(self, model, historical_data, asset):
        for i in range(len(historical_data) - 1):
            input_data = historical_data.iloc[i].values
            next_price = historical_data.iloc[i + 1]['Close']
            prediction = model.predict(input_data.reshape(1, -1))[0]
            
            if prediction > input_data[-1]:  # Simplistic strategy: if predicted price is higher, buy
                self.simulate_trade(asset, amount=1, trade_type='buy', price=input_data[-1], risk_factor=1)
            elif prediction < input_data[-1]:  # If predicted price is lower, sell
                self.simulate_trade(asset, amount=1, trade_type='sell', price=input_data[-1], risk_factor=1)
            
        final_value = self.calculate_portfolio_value()
        print(f"Final portfolio value after backtest: {final_value}")
