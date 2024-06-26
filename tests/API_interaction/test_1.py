import unittest
from unittest.mock import Mock, patch
import asyncio
import aiohttp
import sys

from aiohttp import ClientSession, ClientResponse
from API_interaction import AlphaVantageAPI, PolygonIOAPI, NASDAQAPI


async def main():

    # Create a test configuration dictionary with API keys and base URLs
    class TestAPIs(unittest.TestCase):
        # Test API keys
        test_config = {
            'API': {
                'alphavantage': 'NUZ7HO2VBZZO3GND',
                'polygonio': 'ruqNOBWgLAXuiUM0ugL5WmxbkIdlELp4',
                'nasdaq': '5hSXmst5GSPX2F2VauxN',
                'Finnhub': 'ckuqs6pr01qmtr8lh750ckuqs6pr01qmtr8lh75g',
            }
        }

        @classmethod
        def setUpClass(cls):
            # Mock the ClientSession for aiohttp to avoid actual HTTP requests
            cls.session_mock = Mock(spec=ClientSession)
            cls.response_mock = Mock(spec=ClientResponse)

        def setUp(self):
            # Create instances of the API classes with test API keys and base URLs
            self.alpha_vantage = AlphaVantageAPI("https://www.alphavantage.co/query", self.test_config['API']['alphavantage'])
            self.polygon_io = PolygonIOAPI("https://api.polygon.io/v2", self.test_config['API']['polygonio'])
            self.nasdaq = NASDAQAPI("https://api.nasdaq.com/api", self.test_config['API']['nasdaq'])

        @patch('API_interaction.configparser.ConfigParser', return_value=Mock(**{'read.return_value': test_config}))
        def test_constructor(self, mock_config_parser):
            # Test if the API classes are constructed with the correct attributes
            self.assertEqual(self.alpha_vantage.base_url, "https://www.alphavantage.co/query")
            self.assertEqual(self.alpha_vantage.api_key, "NUZ7HO2VBZZO3GND")
            self.assertEqual(self.polygon_io.base_url, "https://api.polygon.io/v2")
            self.assertEqual(self.polygon_io.api_key, "ruqNOBWgLAXuiUM0ugL5WmxbkIdlELp4")
            self.assertEqual(self.nasdaq.base_url, "https://api.nasdaq.com/api")
            self.assertEqual(self.nasdaq.api_key, "5hSXmst5GSPX2F2VauxN")

        @patch('API_interaction.configparser.ConfigParser', return_value=Mock(**{'read.return_value': test_config}))
        def test_construct_url(self, mock_config_parser):
            # Test the _construct_url method of AlphaVantageAPI
            url_av = self.alpha_vantage._construct_url("AAPL", "daily")
            self.assertEqual(url_av, "https://www.alphavantage.co/query/time_series/daily/AAPL?apikey=NUZ7HO2VBZZO3GND")

        @patch('API_interaction.configparser.ConfigParser', return_value=Mock(**{'read.return_value': test_config}))
        def test_async_fetch_data(self, mock_config_parser):
            # Mock the aiohttp ClientSession and ClientResponse for testing
            self.session_mock.get.return_value.__aenter__.return_value = self.response_mock
            self.response_mock.status = 200
            self.response_mock.json.return_value = {"test_data": "example"}

            # Test the async_fetch_data method of AlphaVantageAPI
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(self.alpha_vantage.async_fetch_data("AAPL", "daily"))
            self.assertEqual(result, {"test_data": "example"})

        @patch('API_interaction.configparser.ConfigParser', return_value=Mock(**{'read.return_value': test_config}))
        def test_async_fetch_data_rate_limit(self, mock_config_parser):
            # Mock the aiohttp ClientSession and ClientResponse for testing rate limiting
            with patch.object(aiohttp.ClientSession, '__aenter__') as mock_aenter:
                with patch.object(aiohttp.ClientSession, '__aexit__') as mock_aexit:
                    mock_aenter.return_value.__aenter__.return_value = self.response_mock
                    mock_aexit.return_value.__aexit__.return_value = None

                    self.response_mock.status = 429

                    # Mock handle_rate_limit to return a test result
                    self.alpha_vantage.handle_rate_limit = Mock(return_value={"test_data": "rate_limit_example"})

                    # Test async_fetch_data when a rate limit is encountered
                    loop = asyncio.get_event_loop()
                    result = loop.run_until_complete(self.alpha_vantage.async_fetch_data("AAPL", "daily"))
                    self.assertEqual(result, {"test_data": "rate_limit_example"})

        @patch('API_interaction.configparser.ConfigParser', return_value=Mock(**{'read.return_value': test_config}))
        def test_async_fetch_data_error_handling(self, mock_config_parser):
            # Mock the aiohttp ClientSession and ClientResponse for testing client errors
            self.session_mock.get.return_value.__aenter__.side_effect = aiohttp.ClientError("Test error")

            # Test async_fetch_data when a client error occurs
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(self.alpha_vantage.async_fetch_data("AAPL", "daily"))
            self.assertIsNone(result)  # Check that None is returned on error

        @patch('API_interaction.configparser.ConfigParser', return_value=Mock(**{'read.return_value': test_config}))
        def test_async_fetch_data_success(self, mock_config_parser):
            # Mock the aiohttp ClientSession and ClientResponse for successful response
            with patch.object(aiohttp.ClientSession, '__aenter__') as mock_aenter:
                with patch.object(aiohttp.ClientSession, '__aexit__') as mock_aexit:
                    mock_aenter.return_value.__aenter__.return_value = self.response_mock
                    mock_aexit.return_value.__aexit__.return_value = None

                    self.response_mock.status = 200
                    self.response_mock.json.return_value = {"test_data": "success_example"}

                    # Test async_fetch_data for a successful response
                    loop = asyncio.get_event_loop()
                    result = loop.run_until_complete(self.alpha_vantage.async_fetch_data("AAPL", "daily"))
                    self.assertEqual(result, {"test_data": "success_example"})

if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Run main() to get the TestAPIs class
    TestAPIs = loop.run_until_complete(main())

    # Load the test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAPIs)

    # Run the test suite
    unittest.TextTestRunner().run(suite)

    loop.close()
