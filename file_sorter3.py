#file_sorter3.py

import os

# Define the directory in which the script is running
directory = os.path.dirname(os.path.abspath(__file__))

# Check for the presence of 'power_models' folder and create it if not present
power_models_dir = os.path.join(directory, 'power_models')
if not os.path.exists(power_models_dir):
    os.makedirs(power_models_dir)

# Check for the presence of 'standard_models' folder and create it if not present
standard_models_dir = os.path.join(directory, 'standard_models')
if not os.path.exists(standard_models_dir):
    os.makedirs(standard_models_dir)

# Iterate over the files in the directory
for filename in os.listdir(directory):
    # Check if it's a file
    if os.path.isfile(os.path.join(directory, filename)):
        # Check for 'power_model' in the filename
        if 'power_model' in filename:
            os.rename(os.path.join(directory, filename), os.path.join(power_models_dir, filename))
        # Check for 'standard_model' in the filename
        elif 'standard_model' in filename:
            os.rename(os.path.join(directory, filename), os.path.join(standard_models_dir, filename))

print("Files sorted!")
