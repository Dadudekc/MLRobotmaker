    def preprocess_new_data(self, new_data):
        scaled_new_data = self.feature_scaler.transform(new_data)
        X_new = self.create_sequence(scaled_new_data, self.lookback)
        return X_new

    async def async_preprocess_data(self):
        """
        Loads and preprocesses data asynchronously, handling errors.
        """
        try:
            # Replace with your actual data loading and preprocessing logic
            await asyncio.sleep(1)  # Placeholder for preprocessing
            # Example: X_train, X_val, y_train, y_val = ...
            
            # Ensure variables are not None
            if None in [X_train, X_val, y_train, y_val]:
                raise ValueError("One or more data variables are None")

            return X_train, X_val, y_train, y_val

        except Exception as e:
            self.display_message(f"Error during data preprocessing: {str(e)}", level="ERROR")
            raise  # Re-raise the exception to be handled by the caller

   
    def neural_network_preprocessing(self, data, scaler_type, close_price_column='close', file_path=None):
        """
        Specific preprocessing for neural network models.

        Args:
            data (DataFrame): The dataset to preprocess.
            scaler_type (str): Type of scaler to use for feature scaling.
            close_price_column (str): Name of the column for close prices, which is the target.
            file_path (str, optional): Path of the data file for logging purposes.

        Returns:
            tuple: Preprocessed features (X) and target values (y).
        """
        try:
            if close_price_column not in data.columns:
                raise ValueError(f"'{close_price_column}' column not found in the data.")

            # Convert date column to numeric features
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'], errors='coerce')
                data['year'] = data['date'].dt.year
                data['month'] = data['date'].dt.month
                data['day'] = data['date'].dt.day
                data.drop(columns=['date'], inplace=True)

            # Prepare your features and target
            X = data.drop(columns=[close_price_column])
            y = data[close_price_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Handle non-numeric columns here (if any)

            # Scaling features
            scaler = self.get_scaler(scaler_type)
            X_scaled = scaler.fit_transform(X)

            return X_scaled, y

        except Exception as e:
            error_message = f"Error in neural network preprocessing: {str(e)}"
            if file_path:
                error_message += f" File path: {file_path}"
            self.utils.log_message(error_message, self, self.log_text, self.is_debug_mode)
            return None, None
        
    def load_unseen_test_data(filepath):
        """
        Load unseen test data from a CSV file.

        Parameters:
        filepath (str): The path to the CSV file containing the test data.

        Returns:
        DataFrame: The loaded test data.
        """
        try:
            data = pd.read_csv(filepath)
            return data
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            return None
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")
            return None
        
    def prepare_and_train_lstm_model(self, df, scaler_type, lookback=60, epochs=50, batch_size=32):
        # Ensure target_column is in the DataFrame
        target_column = 'close'
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        # Convert date to a numerical feature, for example, days since a fixed date
        df['date'] = pd.to_datetime(df['date'])
        reference_date = pd.to_datetime('2000-01-01')
        df['days_since'] = (df['date'] - reference_date).dt.days

        # Exclude the original date column and use 'days_since' for training
        features = df.drop(columns=[target_column, 'date']).values
        target = df[target_column].values

        # Scaling the features and target
        feature_scaler = self.get_scaler(scaler_type)
        scaled_features = feature_scaler.fit_transform(features)
        target = target.reshape(-1, 1)
        target_scaler = self.get_scaler(scaler_type)
        scaled_target = target_scaler.fit_transform(target)

        # Creating sequences for LSTM
        X, y = self.create_sequence(scaled_features, scaled_target.flatten(), lookback)

        # Splitting dataset into training, testing, and new data
        X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size=0.4, random_state=42)
        X_test, X_new, y_test, y_new = train_test_split(X_remaining, y_remaining, test_size=0.5, random_state=42)

        # Defining the LSTM model
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])

        # Compiling the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Training the model
        trained_model = self.train_neural_network_or_lstm(X_train, y_train, X_test, y_test, model_type, epochs)


        return model, feature_scaler, target_scaler, X_test, y_test, X_new, y_new

    def create_sequence(self, features, target, lookback):
        X, y = [], []
        for i in range(lookback, len(features)):
            X.append(features[i-lookback:i])
            y.append(target[i])
        return np.array(X), np.array(y)

    def show_epochs_input(self, event):
        selected_model_type = self.model_type_var.get()
        
        if selected_model_type in ["neural_network", "LSTM"]:
            if not hasattr(self, 'epochs_label'):
                self.epochs_label = tk.Label(self, text="Epochs:")
                self.epochs_entry = tk.Entry(self)

            self.epochs_label.pack(in_=self)
            self.epochs_entry.pack(in_=self)

            self.window_size_label.pack()
            self.window_size_entry.pack()
        else:
            if hasattr(self, 'epochs_label'):
                self.epochs_label.pack_forget()
                self.epochs_entry.pack_forget()
                self.window_size_label.pack_forget()
                self.window_size_entry.pack_forget()

    def load_pretrained_model(self, model_path, custom_objects=None):
        """
        Load a pre-trained model from the specified path.
        
        Args:
            model_path (str): Path to the pre-trained model.
            custom_objects (dict): Optionally, a dictionary of custom objects if the model uses any.
        
        Returns:
            keras.models.Model: The loaded pre-trained model.
        """
        try:
            model = load_model(model_path, custom_objects=custom_objects)
            print("Pre-trained model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading the pre-trained model: {e}")
            return None
        
    def scale_features(self, X, scaler_type):
        """
        Scale the features using the specified scaler, handling both DataFrames and NumPy arrays.

        Args:
            X (DataFrame or numpy array): The feature matrix.
            scaler_type (str): Type of scaler to use for feature scaling.

        Returns:
            DataFrame or numpy array: Scaled feature matrix.
        """
        # Define scalers
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'normalizer': Normalizer(),
            'maxabs': MaxAbsScaler()
        }

        # Select scaler
        scaler = scalers.get(scaler_type, StandardScaler())

        if isinstance(X, pd.DataFrame):
            # If DataFrame, retain column names
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        else:
            # If NumPy array, just scale it
            X_scaled = scaler.fit_transform(X)

        return X_scaled

    def save_scaler(self, scaler, file_path=None):
        try:
            # If file_path is not provided, open a file dialog for the user to select a save location
            if file_path is None:
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".pkl",
                    filetypes=[("Pickle files", "*.pkl"), ("All Files", "*.*")],
                    title="Save Scaler As"
                )

                # Check if the user canceled the save operation
                if not file_path:
                    self.display_message("Save operation canceled.", level="INFO")
                    return

            # Determine the appropriate method to save the scaler
            if isinstance(scaler, (StandardScaler, MinMaxScaler, RobustScaler, Normalizer, MaxAbsScaler)):
                joblib.dump(scaler, file_path)
            else:
                # If scaler is not a recognized type, use pickle
                with open(file_path, 'wb') as file:
                    pickle.dump(scaler, file)

            self.display_message(f"Scaler saved successfully at {file_path}", level="INFO")

        except Exception as e:
            self.display_message(f"Error saving scaler: {str(e)}", level="ERROR")

    def visualize_data(self, data, column_name=None):
        # Validate if the column_name is provided and exists in the DataFrame
        if column_name and column_name in data.columns:
            fig = px.histogram(data, x=column_name)
        else:
            # If column_name is not provided or doesn't exist, default to a known column or handle the error
            # For demonstration, let's default to visualizing the 'close' column
            default_column = 'close'  # Ensure this column exists in your DataFrame
            if default_column in data.columns:
                fig = px.histogram(data, x=default_column)
            else:
                # Handle the case where the default column also doesn't exist
                print("The specified column for visualization does not exist in the DataFrame.")
                return  # Exit the function if the column doesn't exist
            
    def toggle_advanced_settings(self):
        if self.advanced_settings_var.get():
            # Show settings by packing frame and entries
            self.settings_frame.pack(pady=(0, 10))
            self.optimizer_label.pack()
            self.optimizer_entry.pack()
            self.regularization_label.pack()
            self.regularization_entry.pack()
            self.learning_rate_label.pack()
            self.learning_rate_entry.pack()
            self.batch_size_label.pack()
            self.batch_size_entry.pack()
            # Now just pack the already initialized window size widgets
            self.window_size_label.pack()
            self.window_size_entry.pack()
        else:
            # Hide the entire frame to make all settings invisible
            self.settings_frame.pack_forget()
            # Optionally reset the advanced settings variable to False if needed
            self.advanced_settings_var.set(False)


            ============================================================================================================================




































































