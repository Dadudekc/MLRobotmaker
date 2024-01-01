#sample config

[Settings]
# Toggle debug mode (true/false)
DebugMode = false

[Data]
# Default path for data file
DataFilePath = 

[ModelTraining]
# Default type of scaler
DefaultScaler = standard

# Default type of machine learning model
DefaultModelType = linear_regression

# Default number of epochs for neural network training
DefaultEpochs = 10

[Paths]
# Path for saving trained models
ModelSavePath = ./models/

# Path for saving logs
LogPath = ./logs/

[Visualization]
# Default settings for plot appearance
PlotSize = (8, 6)
AlphaValue = 0.5
LineStyle = k--
LineWidth = 2

[Performance]
# Settings related to performance optimizations
ThreadCount = 4

[SAVE_PATH_SECTION]
save_path_dir = /path/to/save/directory

# config.ini

[API_KEYS]
alphavantage = YOUR_ALPHA_VANTAGE_API_KEY
polygonio = 
nasdaq = 
Finnhub = 

[SETTINGS]
DebugMode = False
# Add other settings as needed
