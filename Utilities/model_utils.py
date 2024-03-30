#model_utils.py

import asyncio
from sklearn.metrics import mean_squared_error, r2_score
from shared_data_store import SharedDataStore, Observer
import joblib
import tensorflow as tf
import os

# Assume shared_data_store is initialized somewhere globally
shared_data_store = SharedDataStore()

class ModelUtils:
    def __init__(self):
        # Initialize any necessary attributes here
        pass

    async def calculate_metrics(self, y_true, y_pred):
        """
        Asynchronously calculate and return model evaluation metrics.
        """
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        return {"RMSE": rmse, "R^2": r2}

    async def save_model(self, model, model_name, file_extension=".pkl"):
        """
        Asynchronously save the model in the specified format.
        """
        supported_extensions = {".pkl": self._save_pkl, ".h5": self._save_h5}
        save_func = supported_extensions.get(file_extension)

        if save_func:
            await save_func(model, model_name)
        else:
            print(f"File extension {file_extension} not supported.")

    async def _save_pkl(self, model, model_name):
        """
        Save a model in pickle format.
        """
        filename = f"{model_name}.pkl"
        joblib.dump(model, filename)
        print(f"Model saved as {filename}")

    async def _save_h5(self, model, model_name):
        """
        Save a Keras model in H5 format.
        """
        filename = f"{model_name}.h5"
        model.save(filename)
        print(f"Model saved as {filename}")

    async def train_and_update_store(self, dataset_name, model_name, train_func, **train_params):
        """
        Train a model and update the SharedDataStore with the new model.
        """
        dataset = shared_data_store.get_dataset(dataset_name)
        if dataset:
            model = await train_func(dataset['data'], **train_params)
            shared_data_store.add_model(model_name, model, metadata={'model_type': 'custom'})
        else:
            print(f"Dataset {dataset_name} not found.")

# Example observer that acts upon model updates
class ModelTrainingNotifier(Observer):
    async def update(self, message, event_type):
        print(f"Notification: {message}")

# Register the observer
model_notifier = ModelTrainingNotifier()
shared_data_store.register_observer(model_notifier, interest="model_update")

# Simplified example of asynchronous model training logic
async def train_model_async(dataset, epochs=10):
    # This is a placeholder for actual model training logic. It should be replaced with your specific model training code.
    # For demonstration, let's assume a TensorFlow Keras model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(dataset.shape[1]-1,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Simulate training
    await asyncio.sleep(1)  # Replace this with actual training logic

    return model

model_utils = ModelUtils()

# Example to train a model and save it based on its type
async def example_workflow():
    await model_utils.train_and_update_store("example_dataset", "example_model", train_model_async, epochs=5)
    model_info = shared_data_store.get_model("example_model")
    if model_info:
        model = model_info['model']
        metadata = model_info.get('metadata', {})
        file_extension = ".h5" if metadata.get('model_type') == 'keras' else ".pkl"
        await model_utils.save_model(model, "example_model", file_extension)

# Assuming an event loop is running elsewhere
# asyncio.run(example_workflow())
