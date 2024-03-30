# main.py

from config import Config
from logging_utils import setup_logging

# Initialize configuration
config = Config()

# Setup logging
setup_logging(log_level=config.LOGGING_CONFIG['log_level'], log_file_path=config.LOGGING_CONFIG['file_path'])

# Your application's main logic starts here
if __name__ == "__main__":
    # Example usage of configuration and logging
    print(f"Data path: {config.DATA_PATH}")
    print(f"Model save path: {config.MODEL_SAVE_PATH}")
