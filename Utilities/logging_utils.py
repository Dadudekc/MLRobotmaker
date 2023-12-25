import logging
import os

def setup_logging(config):
    """
    Set up logging configuration.

    Args:
        level (int): Logging level, e.g., logging.INFO, logging.DEBUG.
        log_file (str, optional): Path to the log file. If not provided, logs will be output to console.
    """
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    log_datefmt = '%Y-%m-%d %H:%M:%S'

    # Extract the logging level from the config
    level = config.get('Logging', 'level', fallback='INFO')

    logging.basicConfig(format=log_format, datefmt=log_datefmt, level=level)


def log_info(message):
    """
    Log an informational message.

    Args:
        message (str): Message to be logged.
    """
    logging.info(message)

def log_warning(message):
    """
    Log a warning message.

    Args:
        message (str): Message to be logged.
    """
    logging.warning(message)

def log_error(message):
    """
    Log an error message.

    Args:
        message (str): Message to be logged.
    """
    logging.error(message)

def log_debug(message):
    """
    Log a debug message.

    Args:
        message (str): Message to be logged.
    """
    logging.debug(message)

# Example usage
if __name__ == "__main__":
    setup_logging()  # Call this at the beginning of your main application
    log_info("This is an info message.")
