#logging_utils.py

import logging
import os
from datetime import datetime

# Define a function to set up the logging environment
def setup_logging(log_level=logging.INFO, log_file_path=None):
    """
    Set up logging configuration. Logs will be written to the console and optionally to a file.

    Parameters:
    - log_level: Minimum level of messages to log. Defaults to logging.INFO.
    - log_file_path: Optional path to a log file where logs will be written.
    """
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Define log format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Check if log file path is provided
    if log_file_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        # Create file handler which logs even debug messages
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

# Function to log a message with a specific level
def log_message(message, level=logging.INFO):
    """
    Log a message with the specified severity level.

    Parameters:
    - message: The message to log.
    - level: The severity level of the log. Defaults to logging.INFO.
    """
    if level == logging.DEBUG:
        logging.debug(message)
    elif level == logging.INFO:
        logging.info(message)
    elif level == logging.WARNING:
        logging.warning(message)
    elif level == logging.ERROR:
        logging.error(message)
    elif level == logging.CRITICAL:
        logging.critical(message)
    else:
        logging.info(message)

# Example usage
if __name__ == "__main__":
    # Set up logging to file and console with a specific format and log level
    log_dir = "logs"
    log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
    setup_logging(log_file_path=os.path.join(log_dir, log_filename))
    
    # Log some messages
    log_message("This is an info message.")
    log_message("This is a warning message.", logging.WARNING)
    log_message("This is an error message.", logging.ERROR)
