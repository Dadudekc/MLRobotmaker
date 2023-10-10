import os
import shutil

def move_processed_files(source_dir, destination_dir):
    # Check if destination directory exists, if not, create it
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    # Loop through each file in the source directory
    for file_name in os.listdir(source_dir):
        # Check if the file ends with 'processed_data.csv'
        if file_name.endswith('processed_data.csv'):
            # Construct full file paths for source and destination
            source_path = os.path.join(source_dir, file_name)
            destination_path = os.path.join(destination_dir, file_name)
            
            # Move the file from source to destination
            shutil.move(source_path, destination_path)
            print(f"Moved {file_name} to {destination_dir}")

if __name__ == "__main__":
    source_path = r"C:\Users\Dagurlkc\OneDrive\Desktop\DaDudeKC\MLRobot\csv_files\format1"
    destination_path = r"C:\Users\Dagurlkc\OneDrive\Desktop\DaDudeKC\MLRobot\csv_files\format1\f1_processed_data"
    
    move_processed_files(source_path, destination_path)

def move_processed_files(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith("processed_data.csv"):
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, filename)
            shutil.move(source_file, target_file)
            print(f"Moved {filename} to {target_dir}")

if __name__ == "__main__":
    source_path = "C:\\Users\\Dagurlkc\\OneDrive\\Desktop\\DaDudeKC\\MLRobot\\csv_files\\format2"
    target_path = "C:\\Users\\Dagurlkc\\OneDrive\\Desktop\\DaDudeKC\\MLRobot\\csv_files\\format2\\f2_processed_data"
    move_processed_files(source_path, target_path)
