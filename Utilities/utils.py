import os
import configparser
import logging

def load_configuration(config_file='config.ini'):
    """
    Load a configuration file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        ConfigParser: The configuration object.
    """
    config = configparser.ConfigParser()
    if not os.path.exists(config_file):
        logging.error(f"Configuration file does not exist: {config_file}")
        raise FileNotFoundError(f"Configuration file does not exist: {config_file}")
    config.read(config_file)
    logging.info("Configuration file loaded successfully.")
    return config

def setup_logging(level=logging.INFO):
    """
    Set up basic logging configuration.

    Args:
        level (int): Logging level.
    """
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=level)

def validate_section_keys(config, section, required_keys):
    """
    Validate that all necessary keys are present in a specified configuration section.

    Args:
        config (ConfigParser object): Configuration object to validate.
        section (str): Section name to check in the configuration.
        required_keys (list): Required keys to check for in the section.

    Raises:
        ValueError: If any required keys are missing.
    """
    missing_keys = [key for key in required_keys if key not in config[section]]
    if missing_keys:
        raise ValueError(f"Missing required config key(s) in '{section}': {', '.join(missing_keys)}")

# Additional helper functions can be added here as needed