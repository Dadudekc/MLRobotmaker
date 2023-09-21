import os
import pandas as pd  # Import pandas library
import shutil




# Main function to manage CSV files
def manage_csv_files(csv_directory):
    # Create 'format1' and 'format2' folders if they don't exist
    format1_folder = os.path.join(csv_directory, 'format1')
    format2_folder = os.path.join(csv_directory, 'format2')
    if not os.path.exists(format1_folder):
        os.makedirs(format1_folder)
    if not os.path.exists(format2_folder):
        os.makedirs(format2_folder)

    # Iterate through CSV files in the main directory
    for filename in os.listdir(csv_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(csv_directory, filename)

            # Load the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Check if the DataFrame matches 'format1' criteria
            format1_columns = ['v', 'vw', 'o', 'c', 'h', 'l', 't', 'n']
            if all(col in df.columns for col in format1_columns):
                destination_folder = format1_folder
            # Check if the DataFrame matches 'format2' criteria
            elif all(col in df.columns for col in ['date', 'open', 'high', 'low', 'close', 'volume']):
                destination_folder = format2_folder
            else:
                print(f"Unknown format for file: {filename}")
                continue

            # Move the file to the appropriate folder
            destination_path = os.path.join(destination_folder, filename)
            shutil.move(file_path, destination_path)
            print(f"File {filename} moved to {destination_folder}")

# Main function
def main():
    # Define the directory to save CSV files
    csv_directory = 'csv_files'  # Change this to your desired directory

    # Check if directory exists, create if not
    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)

    # Call the function to manage CSV files
    manage_csv_files(csv_directory)

if __name__ == "__main__":
    main()
