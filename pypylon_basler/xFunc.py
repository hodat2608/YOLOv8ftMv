import tkinter as tk
from tkinter import Label, Entry, Button, Frame
from pypylon import pylon
import cv2
import numpy as np
import time
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtWidgets, QtGui

class Basler_Pylon:
    def __init__(self,mainWindow,ui):
        self.mainWindow=mainWindow
        self.ui = ui
        self.camera = None

    def initialize_camera(self):
        self.devices = pylon.TlFactory.GetInstance().EnumerateDevices()
        listDevice = []
        self.ui.ComboDevices.clear()
        for index, device in enumerate(self.devices):
            device_info = f"[{str(index)}] USB 3.0: {device.GetModelName()} {device.GetSerialNumber()}"
            listDevice.append(device_info)
        self.ui.ComboDevices.addItems(listDevice)
        self.ui.ComboDevices.setCurrentIndex(0)

    def open_device(self):
        # if self.isOpen:
        #     QMessageBox.warning(self.mainWindow, "Error", 'Camera is Running!', QMessageBox.Ok)
        selected_index = self.ui.ComboDevices.currentIndex()
        if selected_index == -1 or len(self.devices) == 0:
            QMessageBox.warning(self.mainWindow, "Error", "No device selected or available.", QMessageBox.Ok)
            return
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice(self.devices[selected_index]))
        self.camera.Open()
        self.ExportValue()
        print(f"Camera {self.devices[selected_index].GetModelName()} with serial {self.devices[selected_index].GetSerialNumber()} opened successfully.")

    def SetValuePrams(self):
        width = self.ui.edtwidth.text()
        height = self.ui.edtheight.text()
        OffsetX = self.ui.edtOffsetX.text()
        OffsetY = self.ui.edtOffsetY.text()
        exposure_time = self.ui.edtExposureTime.text()
        gain = self.ui.edtGain.text()
        AcquisitionFrameRate = self.ui.edtFrameRate.text()
        UserSet = self.ui.combo_usersetload.currentText()
        self.camera.UserSetSelector.SetValue(str(UserSet))
        self.camera.UserSetLoad.Execute()
        self.camera.Width.SetValue(int(float(width)))
        self.camera.Height.SetValue(int(float(height)))
        self.camera.OffsetX.SetValue(int(float(OffsetX)))
        self.camera.OffsetY.SetValue(int(float(OffsetY)))
        self.camera.ExposureTime.SetValue(float(exposure_time))
        self.camera.Gain.SetValue(float(gain))
        self.camera.AcquisitionFrameRate.SetValue(int(float(AcquisitionFrameRate)))

    def ExportValue(self):
        self.ui.combo_usersetload.findText(self.camera.UserSetSelector.GetValue())
        self.ui.edtwidth.setText("{0:.2f}".format(self.camera.Width.GetValue()))
        self.ui.edtheight.setText("{0:.2f}".format(self.camera.Height.GetValue()))
        self.ui.edtOffsetX.setText("{0:.2f}".format(self.camera.OffsetX.GetValue()))
        self.ui.edtOffsetY.setText("{0:.2f}".format(self.camera.OffsetY.GetValue()))
        self.ui.edtExposureTime.setText("{0:.2f}".format(self.camera.ExposureTime.GetValue()))
        self.ui.edtGain.setText("{0:.2f}".format(self.camera.Gain.GetValue()))
        self.ui.edtFrameRate.setText("{0:.2f}".format(self.camera.AcquisitionFrameRate.GetValue()))

    def Trigger(self):
        startTime = time.time()
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grab_result.GrabSucceeded():
            image = grab_result.GetArray()
            grab_result.Release()
            self.camera.StopGrabbing()
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image_rgb.shape
            bytes_per_line = 3 * width
            q_img = QtGui.QImage(image_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_img)
            self.ui.labelDisplay.setPixmap(pixmap)
            runningTime = (time.time() - startTime)

    def start_camera_stream(self):
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        while self.camera.IsGrabbing():
            grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                image = grab_result.GetArray()
                grab_result.Release()
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width, channel = image_rgb.shape
                bytes_per_line = 3 * width
                q_img = QtGui.QImage(image_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(q_img)
                self.ui.labelDisplay.setPixmap(pixmap)
                QtWidgets.QApplication.processEvents()
        
    def stop_camera_stream(self):
        self.camera.StopGrabbing()

    def close_device(self):
        if self.isOpen:
            self.camera.DestroyDevice()
            self.isOpen = False
        self.isGrabbing = False
        self.enable_controls()

    


