from data_preprocessing import preprocess_data
from model_training import train_model, evaluate_model, save_model
import sys
sys.path.append('C:\\Users\\Dagurlkc\\OneDrive\\Desktop\\DaDudeKC\\MLRobot')
from Utilities.Utils import load_configuration, setup_logging
import statsmodels
import os
import pandas as pd
import configparser

def main():
    # Set up logging
    setup_logging()

    # Load configuration
    config_path = 'config.ini'
    config = configparser.ConfigParser()
    config.read(config_path)

    # Example of using the configuration
    data_folder = config['Paths']['data_folder']
    models_directory = config['Paths']['models_directory']

    # Sample workflow (assuming CSV files and model type are predefined)
    for csv_file in os.listdir(data_folder):
        file_path = os.path.join(data_folder, csv_file)
        if csv_file.endswith('.csv'):
            # Read and preprocess data
            data = pd.read_csv(file_path)
            preprocessed_data = preprocess_data(data)

            # Debugging: Check if 'target' column exists
            if 'target' not in preprocessed_data.columns:
                print(f"Target column not found in {csv_file}. Available columns: {preprocessed_data.columns}")
                continue  # Skip to the next file

            # Splitting data into features and target
            X = preprocessed_data.drop('target', axis=1)
            y = preprocessed_data['target']

            # Train the model (example with random forest)
            model = train_model(X, y, model_type='random_forest')

            # Evaluate the model
            performance = evaluate_model(model, X_test, y_test)  # Assuming X_test and y_test are defined
            print(f'Performance of the model on {csv_file}: {performance}')

            # Save the model
            model_filename = os.path.join(models_directory, f'{csv_file}_model.pkl')
            save_model(model, model_filename)

if __name__ == "__main__":
    main()
