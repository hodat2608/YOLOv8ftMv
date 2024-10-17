from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import sys
from xFunc import *
from nextimg import *
def main_mvs(): 
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = Ui_MainWindow()
    cfg = Basler_Pylon(mainWindow,ui)
    ui.setupUi(mainWindow)
    ui.bnEnum.clicked.connect(cfg.initialize_camera)
    ui.bnOpen.clicked.connect(cfg.open_device)
    ui.bnSetParam.clicked.connect(cfg.SetValuePrams)
    ui.bnSoftwareTrigger.clicked.connect(cfg.Trigger)
    ui.bnStart.clicked.connect(cfg.start_camera_stream)
    
    mainWindow.show()
    app.exec_()

main_mvs()


