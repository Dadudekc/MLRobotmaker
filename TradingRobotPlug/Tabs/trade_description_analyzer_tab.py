#trade_description_analyzer_tab.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import spacy
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import pipeline

class TradeDescriptionAnalyzerTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.nlp = spacy.load("en_core_web_sm")
        self.create_widgets()
        self.model = None

    def create_widgets(self):
        self.upload_button = ttk.Button(self, text="Upload CSV", command=self.upload_data)
        self.upload_button.pack()

        self.analyze_button = ttk.Button(self, text="Analyze", command=self.analyze_data)
        self.analyze_button.pack()

        self.output_text = tk.Text(self, height=15, width=50)
        self.output_text.pack()

    def upload_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.data = pd.read_csv(file_path)
            messagebox.showinfo("Info", "Data uploaded successfully.")

    def preprocess_data(self):
        encoder = LabelEncoder()
        self.data['Outcome'] = encoder.fit_transform(self.data['Outcome'])
        # Additional preprocessing steps can be added here

    def build_predictive_model(self, X_train, y_train):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=10, batch_size=10)

    def analyze_sentiment(self, description):
        sentiment_pipeline = pipeline("sentiment-analysis")
        return sentiment_pipeline(description)[0]['label']

    def analyze_data(self):
        if self.data is not None:
            self.preprocess_data()
            X_train, X_test, y_train, y_test = train_test_split(self.data['Description'], self.data['Outcome'], test_size=0.3)
            X_train_features = np.array([self.nlp(text).vector for text in X_train])

            self.build_predictive_model(X_train_features, y_train)
            accuracy = self.model.evaluate(X_train_features, y_train)[1]
            self.output_text.insert(tk.END, f"Model trained with accuracy: {accuracy}\n")

            # Example to show usage - this would be expanded upon for full functionality
            new_description = "Example trade description"
            sentiment = self.analyze_sentiment(new_description)
            self.output_text.insert(tk.END, f"Sentiment Analysis of '{new_description}': {sentiment}\n")
        else:
            messagebox.showwarning("Warning", "No data loaded.")
