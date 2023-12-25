#file_management.py

import os
import shutil
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def create_directory(directory_path):
    """
    Create a directory if it does not exist.

    Args:
        directory_path (str): Path of the directory to be created.
    """
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
        logger.info(f"Directory created at: {directory_path}")
    except Exception as e:
        logger.error(f"Error creating directory: {e}")

def save_data_to_csv(data, file_path):
    """
    Save given data to a CSV file.

    Args:
        data (pd.DataFrame): Data to be saved.
        file_path (str): Path of the CSV file to save data.
    """
    try:
        data.to_csv(file_path, index=False)
        logger.info(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to CSV: {e}")

def sort_csv_files(csv_directory, format_specifications):
    """
    Organize CSV files into subdirectories based on their format.

    Args:
        csv_directory (str): Directory containing the CSV files.
        format_specifications (dict): Dictionary specifying the format criteria for each subdirectory.
    """
    for folder, format_columns in format_specifications.items():
        folder_path = os.path.join(csv_directory, folder)
        create_directory(folder_path)

    for filename in os.listdir(csv_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(csv_directory, filename)
            try:
                df = pd.read_csv(file_path)

                # Determine the destination folder based on the format
                for folder, format_columns in format_specifications.items():
                    if all(col in df.columns for col in format_columns):
                        destination_folder = os.path.join(csv_directory, folder)
                        shutil.move(file_path, destination_folder)
                        logger.info(f"File {filename} moved to {destination_folder}")
                        break
            except Exception as e:
                logger.error(f"Error sorting file {filename}: {e}")

# Example usage
if __name__ == "__main__":
    csv_dir = '/path/to/csv_directory'
    formats = {
        'format1': ['v', 'vw', 'o', 'c', 'h', 'l', 't', 'n'],
        'format2': ['date', 'open', 'high', 'low', 'close', 'volume']
    }
    sort_csv_files(csv_dir, formats)
