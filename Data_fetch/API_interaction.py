#API_interaction.py

import asyncio
import aiohttp
import logging
import configparser

# Set up logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


class BaseAPI:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.logger = logging.getLogger(self.__class__.__name__)

    def _construct_url(self, symbol, interval):
        # This method will be overridden in each subclass
        raise NotImplementedError

    async def async_fetch_data(self, symbol, interval):
        # This method will be overridden in each subclass
        raise NotImplementedError

    async def handle_rate_limit(self, retry_after=60, max_retries=5):
        for attempt in range(max_retries):
            await asyncio.sleep(retry_after)
            result = await self.async_fetch_data()
            if result is not None:
                return result
        self.logger.error(f"Max retries reached for {self.__class__.__name__}")
        return None


class AlphaVantageAPI(BaseAPI):
    def _construct_url(self, symbol, interval):
        return f"{self.base_url}/time_series/{interval}/{symbol}?apikey={self.api_key}"

    async def async_fetch_data(self, symbol, interval):
        url = self._construct_url(symbol, interval)
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status == 429:
                        await self.handle_rate_limit()

                    response.raise_for_status()
                    data = await response.json()
                    self.logger.info(f"Data successfully fetched from AlphaVantage for {symbol}")
                    return data
            except aiohttp.ClientError as err:
                self.logger.error(f"An error occurred: {err}")
            return None


class PolygonIOAPI(BaseAPI):
    def _construct_url(self, symbol, interval):
        return f"{self.base_url}/open-close/{symbol}/{interval}?apiKey={self.api_key}"

    async def async_fetch_data(self, symbol, interval):
        url = self._construct_url(symbol, interval)
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status == 429:
                        return await self.handle_rate_limit()

                    response.raise_for_status()
                    data = await response.json()
                    self.logger.info(f"Data successfully fetched from Polygon.io for {symbol}")
                    return data
            except aiohttp.ClientError as err:
                self.logger.error(f"An error occurred: {err}")
            return None


class NASDAQAPI(BaseAPI):
    def _construct_url(self, symbol, interval):
        return f"{self.base_url}/data/{symbol}/{interval}?apikey={self.api_key}"

    async def async_fetch_data(self, symbol, interval):
        url = self._construct_url(symbol, interval)
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status == 429:
                        return await self.handle_rate_limit()

                    response.raise_for_status()
                    data = await response.json()
                    self.logger.info(f"Data successfully fetched from NASDAQ for {symbol}")
                    return data
            except aiohttp.ClientError as err:
                self.logger.error(f"An error occurred: {err}")
            return None


# Load API keys and base URLs from the config.ini file
config = configparser.ConfigParser()
config.read('config.ini')

# Define API keys and base URLs
alpha_vantage_api_key = config['API']['alphavantage']
polygon_io_api_key = config['API']['polygonio']
nasdaq_api_key = config['API']['nasdaq']

alpha_vantage_base_url = 'https://www.alphavantage.co/query'
polygon_io_base_url = 'https://api.polygon.io/v2'
nasdaq_base_url = 'https://api.nasdaq.com/api'

async def main():
    alpha_vantage = AlphaVantageAPI(alpha_vantage_base_url, alpha_vantage_api_key)
    polygon_io = PolygonIOAPI(polygon_io_base_url, polygon_io_api_key)
    nasdaq = NASDAQAPI(nasdaq_base_url, nasdaq_api_key)

    # Fetch data asynchronously (example)
    data_av = await alpha_vantage.async_fetch_data("AAPL", "daily")
    # Add similar calls for PolygonIOAPI and NASDAQAPI

asyncio.run(main())
