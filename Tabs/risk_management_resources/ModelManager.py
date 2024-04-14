import keras.models
import joblib
import pickle
import numpy as np
from tkinter import filedialog

class ModelManager:
    def __init__(self):
        self.model = None

    def load_model(self, callback):
        file_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.h5"), ("Model Files", "*.joblib"), ("Pickle Files", "*.pkl")])
        if not file_path:
            callback("Model loading canceled.")
            return

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
        if self.model:
            try:
                prediction = self.model.predict(np.array([input_data]).reshape(1, -1))
                callback(f"Prediction: {prediction[0]}")
            except Exception as e:
                callback(f"Error making prediction: {e}")
