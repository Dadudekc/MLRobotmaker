import configparser
import os
import logging

# Initialize logging
logger = logging.getLogger(__name__)

def load_config(file_path):
    """
    Load the configuration file.

    Args:
        file_path (str): Path to the configuration file.

    Returns:
        configparser.ConfigParser: The loaded configuration object, or None if loading fails.
    """
    config = configparser.ConfigParser()
    if not os.path.exists(file_path):
        logger.error(f"The configuration file {file_path} was not found.")
        return None

    config.read(file_path)
    return config

def validate_config(config):
    """
    Validate the presence and correctness of required configurations.

    Args:
        config (configparser.ConfigParser): The loaded configuration object.

    Returns:
        bool: True if validation is successful, False otherwise.
    """
    required_settings = {
        'API': ['alphavantage', 'polygonio', 'nasdaq'],
        'Settings': ['csv_directory']
    }

    for section, keys in required_settings.items():
        if not config.has_section(section):
            logger.error(f"Missing section: '{section}' in config file.")
            return False
        for key in keys:
            if not config.has_option(section, key):
                logger.error(f"Missing key: '{key}' in section: '{section}' in config file.")
                return False

    return True


