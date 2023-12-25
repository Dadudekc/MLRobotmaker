import configparser
from pathlib import Path

class ConfigManager:
    def __init__(self, config_file='config.ini'):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.load_config()

    def load_config(self):
        """ Load configuration from the file. """
        self.config.read(self.config_file)

    def save_config(self):
        """ Save the current configuration to the file. """
        with open(self.config_file, 'w') as configfile:
            self.config.write(configfile)

    def get_path(self, section, key, fallback=None):
        """ Retrieve a path setting from the configuration. """
        project_dir = Path(__file__).resolve().parent
        path = self.config.get(section, key, fallback=fallback)
        return project_dir / path

    def set_path(self, section, key, value):
        """ Set a path setting in the configuration. """
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = str(value)

    # Add more getters and setters for different configuration options as needed.