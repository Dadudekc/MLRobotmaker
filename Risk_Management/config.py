from datetime import date  # Import the date function
import configparser

ALPHA_VANTAGE_API_KEY = 'C6AG9NZX6QIPYTX4'
INITIAL_CAPITAL = 100000
COMMISSION = 0.001
SLIPPAGE = 0.001
LEVERAGE = 1

ASSET_VALUES = {
    'AAPL': {
        'id': 1.0,
        'risk_factor': 0.5,
        'market_segment': 'Equities',
        'sector': 'Technology',
        'beta': 1.2,
        'dividend_yield': 1.5,
        'market_cap': '2.4T',
        'options': [
            {'strike_price': 150, 'expiry_date': date(2024, 6, 21), 'option_type': 'call'},
            {'strike_price': 130, 'expiry_date': date(2024, 6, 21), 'option_type': 'put'}
        ],
        'historical_performance': []
    },
    'TSLA': {
        'id': 2.0,
        'risk_factor': 0.7,
        'market_segment': 'Equities',
        'sector': 'Automotive',
        'beta': 1.5,
        'dividend_yield': 0.0,
        'market_cap': '700B',
        'options': [
            {'strike_price': 600, 'expiry_date': date(2024, 6, 21), 'option_type': 'call'}
        ],
        'historical_performance': []
    },
}

def load_config(config_path='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config
