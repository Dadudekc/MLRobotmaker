import fileinput
import os

directory = 'C:/Users/Dagurlkc/OneDrive/Desktop/DaDudeKC/MLRobot'

for root, _, files in os.walk(directory):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            with fileinput.FileInput(filepath, inplace=True) as f:
                for line in f:
                    print(line.replace('learning_rate=', 'learning_rate='), end='')