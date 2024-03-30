# config.py

import os

class Config:
    # Data paths
    DATA_PATH = "path/to/your/dataset.csv"
    MODEL_SAVE_PATH = "path/to/save/your/model/"

    # Model parameters
    MODEL_PARAMS = {
        "neural_network": {
            "layers": [
                {"type": "Dense", "units": 64, "activation": "relu"},
                {"type": "Dense", "units": 1, "activation": "linear"}
            ],
            "optimizer": "adam",
            "epochs": 10,
            "batch_size": 32
        },
        "LSTM": {
            "units": 50,
            "epochs": 10,
            "batch_size": 32
        },
        "lightgbm": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "max_depth": -1
        },
        "xgboost": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6
        }
    }

    # Logging configuration
    LOGGING_CONFIG = {
        "file_path": os.path.join(os.getcwd(), "training_logs.log"),
        "log_level": "INFO"
    }
