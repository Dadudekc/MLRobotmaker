# model_manager.py

import keras.models
import pickle
import joblib
from tkinter import filedialog
import numpy as np

class ModelManager:
    def __init__(self):
        self.model = None

    def load_model(self, callback):
        file_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.h5"), ("Pickle Files", "*.pkl"), ("Joblib Files", "*.joblib")])
        if file_path:
            try:
                if file_path.endswith('.h5'):
                    self.model = keras.models.load_model(file_path)
                elif file_path.endswith('.joblib'):
                    self.model = joblib.load(file_path)
                elif file_path.endswith('.pkl'):
                    with open(file_path, 'rb') as f:
                        self.model = pickle.load(f)
                callback("Model loaded successfully.")
            except Exception as e:
                callback(f"Failed to load model: {e}")

    def predict(self, input_data, callback):
        if self.model is None:
            callback("No model loaded. Please load a model first.")
            return None
        try:
            prediction = self.model.predict(np.array([input_data]).reshape(1, -1))
            callback(f"Prediction: {prediction[0]}")
            return prediction[0]
        except Exception as e:
            callback(f"Error making prediction: {e}")
            return None
