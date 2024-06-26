import functools
import logging

class ModelConfigManager:
    """
    Manages configurations for various models used throughout the project.
    """
    def __init__(self):
        self.model_configs = {
            "neural_network": {
                "epochs": 50,
                "window_size": 30,
                "optimizer": "adam",
                "loss": "mean_squared_error"
            },
            "LSTM": {
                "epochs": 50,
                "window_size": 30,
                "units": 100,
                "dropout": 0.2,
                "recurrent_dropout": 0.2
            },
            "ARIMA": {
                "p_value": 1,
                "d_value": 1,
                "q_value": 1
            },
            "linear_regression": {
                "regularization": 0.01  # Alpha value for Ridge or Lasso regression
            },
            "random_forest": {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1
            }
            # Additional models and their configurations can be added here.
        }

    def get_model_config(self, model_type):
        """
        Retrieve the configuration for a specific model type.

        Args:
            model_type (str): The type of model for which to retrieve the configuration.

        Returns:
            dict: The configuration dictionary for the specified model type, if available.
        """
        return self.model_configs.get(model_type, {})

class ExceptionHandler:
    """
    A class to handle exceptions raised by functions with configurable logging and behavior.
    """
    def __init__(self, log_function=None, re_raise=False, handle_type=Exception, default_value=None):
        self.log_function = log_function
        self.re_raise = re_raise
        self.handle_type = handle_type
        self.default_value = default_value

    def __call__(self, func):
        """
        Allows the class instance to be used as a decorator.

        Args:
            func (callable): The function to decorate.

        Returns:
            callable: The decorated function.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except self.handle_type as e:
                if self.log_function:
                    self.log_function(f"Exception in {func.__name__}: {e}")
                if self.re_raise:
                    raise
                return self.default_value
        return wrapper

# Example usage of ExceptionHandler
def log_error(message):
    logging.error(message)

# Create an exception handler that logs errors and re-raises exceptions
error_handler = ExceptionHandler(log_function=log_error, re_raise=True)
