# model_training.py

# Section 1: Imports and Setup
import logging
import joblib
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from skopt import BayesSearchCV
from keras_tuner import HyperModel
import tensorflow as tf
from tensorflow import keras
from kerastuner.tuners import RandomSearch
import logging
import joblib
from typing import Optional, Any



# Section 2: Model Training Functions
# Function to train a specific type of model
def train_model(X_train, y_train, model_type='linear_regression'):
    """
    Train a machine learning model based on the specified type.

    Args:
        X_train (DataFrame): Training features.
        y_train (Series): Training target.
        model_type (str): Type of model to train.

    Returns:
        model: Trained model object.
    """
    if model_type == 'linear_regression':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor()
    elif model_type == 'neural_network':
        model = create_neural_network(X_train.shape[1])
    else:
        logging.error(f"Unsupported model type: {model_type}")
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.fit(X_train, y_train)
    return model

# Function for creating a neural network model
def create_neural_network(input_shape):
    """
    Create a simple neural network for regression tasks.

    Args:
        input_shape (int): The shape of the input data.

    Returns:
        model: A compiled neural network model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Section 3: Model Evaluation
# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.

    Args:
        model: The trained model to evaluate.
        X_test (DataFrame): Test features.
        y_test (Series): Test target.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return {'mean_squared_error': mse, 'r2_score': r2}

# Section 4: Model Saving
# Function to save the trained model
def save_model(model, filename):
    """
    Save the trained model to a file.

    Args:
        model: The trained model to save.
        filename (str): File path to save the model.

    Returns:
        None
    """
    joblib.dump(model, filename)
    logging.info(f"Model saved to {filename}")

#section 5: Hypertuning

class CustomHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = keras.Sequential()
        
        # Define the number of units in the first dense layer as a hyperparameter
        units = hp.Int('units', min_value=32, max_value=512, step=32)
        model.add(keras.layers.Dense(units=units, activation='relu', input_shape=(self.input_shape,)))
        
        # Add more dense layers or other types of layers as needed
        # model.add(keras.layers.Dense(units=hp.Int('units_2', min_value=32, max_value=512, step=32), activation='relu'))
        
        # Define the learning rate for the optimizer as a hyperparameter
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.add(keras.layers.Dense(1, activation='linear'))
        
        # Compile the model with the defined optimizer and loss function
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        return model

#function to load model    
def load_model(model_file_path: str, raise_exception: bool = False) -> Optional[Any]:
    """
    Load a machine learning model from a file.

    Parameters:
    model_file_path (str): The path to the model file.
    raise_exception (bool): If True, raises an exception on failure. Otherwise, returns None.

    Returns:
    Optional[Any]: The loaded model or None if an error occurs and raise_exception is False.

    Raises:
    Exception: Various exceptions related to file handling or model loading, if raise_exception is True.
    """
    try:
        loaded_model = joblib.load(model_file_path)
        return loaded_model
    except FileNotFoundError:
        logging.error(f"Model file not found: {model_file_path}")
    except joblib.JoblibException as e:
        logging.error(f"Error loading model from {model_file_path}: {str(e)}")
    except Exception as e:
        logging.error(f"Unknown error loading model from {model_file_path}: {str(e)}")

    if raise_exception:
        raise
    else:
        return None

# Function for hyperparameter tuning using Keras Tuner
def perform_hyperparameter_tuning(X_train, y_train, input_shape, max_trials=5, epochs=10):
    hypermodel = CustomHyperModel(input_shape=input_shape)

    tuner = RandomSearch(
        hypermodel,
        objective='val_accuracy',
        max_trials=max_trials,
        directory='my_tuner_dir',
        project_name='my_tuner_project'
    )

    tuner.search(X_train, y_train, epochs=epochs)
    return tuner.get_best_models(num_models=1)[0]


# Function for hyperparameter tuning
def bayesian_hyperparameter_tuning(X_train, y_train, model, search_space, n_iter=50, cv=3):
    """
    Perform hyperparameter tuning using Bayesian Optimization.

    Args:
        X_train (DataFrame): Training features.
        y_train (Series): Training target.
        model: The machine learning model to tune.
        search_space (dict): The hyperparameter search space.
        n_iter (int): Number of iterations for the search.
        cv (int): Number of folds for cross-validation.

    Returns:
        best_model: The best model found by the Bayesian optimizer.
    """
    bayes_cv_tuner = BayesSearchCV(
        estimator=model,
        search_spaces=search_space,
        scoring='neg_mean_squared_error',
        n_iter=n_iter,
        cv=cv,
        n_jobs=-1
    )

    bayes_cv_tuner.fit(X_train, y_train)
    best_model = bayes_cv_tuner.best_estimator_
    return best_model

if __name__ == "__main__":
    import numpy as np
    X_train = np.random.rand(100, 10)  # 100 samples, 10 features
    y_train = np.random.rand(100)
    X_test = np.random.rand(20, 10)  # 20 samples, 10 features
    y_test = np.random.rand(20)

    # For training and evaluating a model
    #model, metrics = train_and_evaluate_model(X_train, y_train, X_test, y_test, 'random_forest')
    #print(f"Model performance: {metrics}")

    # For neural network hyperparameter tuning (example usage)
    # Ensure X_train and y_train are NumPy arrays
    #best_model = perform_hyperparameter_tuning(X_train, y_train, input_shape=X_train.shape[1])
    #print("Best model obtained from hyperparameter tuning:")
    #best_model.summary()
