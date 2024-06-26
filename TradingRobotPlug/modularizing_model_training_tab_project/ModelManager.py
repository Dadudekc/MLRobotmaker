import joblib
import pickle
import logging
import os
from datetime import datetime
from keras.models import load_model as keras_load_model
from sklearn.base import BaseEstimator
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, config, utils):
        self.config = config
        self.utils = utils
        self.supported_models = {
            'linear_regression': ".joblib",
            'random_forest': ".joblib",
            'lstm': ".h5",
            'neural_network': ".h5",
            'arima': ".pkl"
        }

    def save_model(self, model, model_type, file_path):
        model_type = model_type.lower()
        if model_type not in self.supported_models:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")

        file_extension = self.supported_models[model_type]
        if not file_path.endswith(file_extension):
            file_path += file_extension

        try:
            if model_type in ['linear_regression', 'random_forest']:
                joblib.dump(model, file_path)
            elif model_type in ['lstm', 'neural_network']:
                model.save(file_path)
            elif model_type == 'arima':
                with open(file_path, 'wb') as pkl:
                    pickle.dump(model, pkl)
            logger.info(f"Model of type '{model_type}' saved successfully at {file_path}")
        except Exception as e:
            logger.error(f"Failed to save the model due to: {e}")
            raise

    def load_model(self, model_type, file_path):
        model_type = model_type.lower()
        if model_type not in self.supported_models:
            logger.error(f"Unsupported model type for loading: {model_type}")
            raise ValueError(f"Unsupported model type for loading: {model_type}")

        try:
            if model_type in ['linear_regression', 'random_forest']:
                return joblib.load(file_path)
            elif model_type in ['lstm', 'neural_network']:
                return keras_load_model(file_path)
            elif model_type == 'arima':
                with open(file_path, 'rb') as pkl:
                    return pickle.load(pkl)
        except Exception as e:
            logger.error(f"Failed to load the model due to: {e}")
            raise

    def determine_model_type(self, model):
        """Determine the model type based on its class for saving or informational purposes."""
        if isinstance(model, BaseEstimator):
            return 'sklearn_model'
        elif isinstance(model, keras.models.Model):
            return 'keras_model'
        elif isinstance(model, torch.nn.Module):
            return 'torch_model'
        elif "ARIMA" in str(type(model)):  # Assume it's a statsmodels ARIMA
            return 'arima'
        else:
            logger.error("Unknown model type.")
            raise ValueError("Unknown model type.")

    def save_model_metadata(self, model, model_type, file_path):
        """Save additional metadata for a model."""
        metadata = {
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': model_type,
            'file_path': file_path
        }
        metadata_file_path = file_path + '.meta'
        try:
            with open(metadata_file_path, 'w') as f:
                f.write(str(metadata))
            logger.info(f"Metadata for model type '{model_type}' saved successfully at {metadata_file_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata due to: {e}")
            raise

if __name__ == "__main__":
    manager = ModelManager()
    # Example usage: Load a model and print its type
    try:
        model = manager.load_model('random_forest', 'path/to/random_forest.joblib')
        model_type = manager.determine_model_type(model)
        print(f"Loaded model type: {model_type}")
    except Exception as e:
        print(f"Error: {e}")

