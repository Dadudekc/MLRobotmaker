import sys
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QWidget, QPushButton, QFileDialog, QMessageBox
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
import pandas as pd
import joblib

class PredictionAppUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Time Series Prediction with Interactive Chart, Model Loading, and Export Features')
        self.setGeometry(100, 100, 800, 600)
        self.model = None  # Initialize model to None

        # Data initialization
        self.xData = list(range(10))  # Initial X values
        self.yData = [val ** 0.5 for val in self.xData]  # Initial Y values for demonstration

        self.initUI()
        self.initTimer()
    
    def initUI(self):
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)
        
        self.graphWidget = pg.PlotWidget()
        self.main_layout.addWidget(self.graphWidget)
        self.plotData()
        
        self.loadModelButton = QPushButton('Load Model', self)
        self.loadModelButton.clicked.connect(self.loadModel)
        self.main_layout.addWidget(self.loadModelButton)
        
        self.exportDataButton = QPushButton('Export Data', self)
        self.exportDataButton.clicked.connect(lambda: self.exportPredictionData(self.xData, self.yData))
        self.main_layout.addWidget(self.exportDataButton)
        
        self.saveChartButton = QPushButton('Save Chart', self)
        self.saveChartButton.clicked.connect(self.saveChartAsImage)
        self.main_layout.addWidget(self.saveChartButton)

    def initTimer(self):
        self.timer = QTimer()
        self.timer.setInterval(1000)  # Update every second
        self.timer.timeout.connect(self.simulateRealTimeData)
        self.timer.start()

    def plotData(self):
        self.graphWidget.clear()
        self.graphWidget.plot(self.xData, self.yData, pen='r')

    def simulateRealTimeData(self):
        if self.model:
            newX = self.xData[-1] + 1
            newY = self.model.predict(np.array([[newX]]))[0]
            self.updatePlot((newX, newY))
        else:
            print("No model loaded or prediction failed.")

    def updatePlot(self, new_data):
        newX, newY = new_data
        self.xData.append(newX)
        self.yData.append(newY)
        self.plotData()

    def loadModel(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Joblib Model File", "", "Joblib Files (*.joblib);;All Files (*)")
        if fileName:
            try:
                self.model = joblib.load(fileName)
                QMessageBox.information(self, "Model Loaded", "The model has been loaded successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Loading Failed", f"Failed to load model: {e}")

    def exportPredictionData(self, xData, yData):
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Prediction Data", "", "CSV Files (*.csv);;All Files (*)")
        if fileName:
            df = pd.DataFrame({'X': xData, 'Y': yData})
            df.to_csv(fileName, index=False)
            QMessageBox.information(self, "Export Successful", "Data exported successfully.")

    def saveChartAsImage(self):
        exporter = pg.exporters.ImageExporter(self.graphWidget.plotItem)
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Chart As Image", "", "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*)")
        if fileName:
            exporter.export(fileName)
            QMessageBox.information(self, "Export Successful", "Chart saved as image.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = PredictionAppUI()
    mainWindow.show()
    sys.exit(app.exec_())
