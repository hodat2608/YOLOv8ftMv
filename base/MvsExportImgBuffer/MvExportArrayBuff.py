# -- coding: utf-8 --

from PyQt5.QtWidgets import *
from base.MvsExportImgBuffer.CamOperation_class import CameraOperation
from base.MvsExportImgBuffer.MvCameraControl_class import *
from base.MvsExportImgBuffer.MvErrorDefine_const import *
from base.MvsExportImgBuffer.CameraParams_header import *
from base.MvsExportImgBuffer.PyUICBasicDemo import Ui_MainWindow


# 获取选取设备信息的索引，通过[]之间的字符去解析
def TxtWrapBy(start_str, end, all):
    start = all.find(start_str)
    if start >= 0:
        start += len(start_str)
        end = all.find(end, start)
        if end >= 0:
            return all[start:end].strip()


# 将返回的错误码转换为十六进制显示
def ToHexStr(num):
    chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    hexStr = ""
    if num < 0:
        num = num + 2 ** 32
    while num >= 16:
        digit = num % 16
        hexStr = chaDic.get(digit, str(digit)) + hexStr
        num //= 16
    hexStr = chaDic.get(num, str(num)) + hexStr
    return hexStr


class LoadDiviceEnvCam():
    def __init__(self,n_numcamera):
        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        self.cam = MvCamera()
        self.nSelCamIndex = n_numcamera
        self.obj_cam_operation = 0
        self.isOpen = False
        self.isGrabbing= False
        self.isCalibMode = False # 是否是标定模式（获取原始图像）
   
    # 绑定下拉列表至设备信息索引
    def xFunc(event):
        # global nSelCamIndex
        # nSelCamIndex = TxtWrapBy("[", "]", ui.ComboDevices.get())
        pass

    # ch:枚举相机 | en:enum devices
    def enum_devices(self):

        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, self.deviceList)
        if ret != 0:
            strError = "Enum devices fail! ret = :" + ToHexStr(ret)
            # QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
            return ret

        if self.deviceList.nDeviceNum == 0:
            # QMessageBox.warning(mainWindow, "Info", "Find no device", QMessageBox.Ok)
            return ret
        print("Find %d devices!" % self.deviceList.nDeviceNum)

        devList = []
        for i in range(0, self.deviceList.nDeviceNum):
            mvcc_dev_info = cast(self.deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                print("\ngige device: [%d]" % i)
                chUserDefinedName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chUserDefinedName:
                    if 0 == per:
                        break
                    chUserDefinedName = chUserDefinedName + chr(per)
                print("device user define name: %s" % chUserDefinedName)

                chModelName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                    if 0 == per:
                        break
                    chModelName = chModelName + chr(per)

                print("device model name: %s" % chModelName)

                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
                devList.append(
                    "[" + str(i) + "]GigE: " + chUserDefinedName + " " + chModelName + "(" + str(nip1) + "." + str(
                        nip2) + "." + str(nip3) + "." + str(nip4) + ")")
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                print("\nu3v device: [%d]" % i)
                chUserDefinedName = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chUserDefinedName:
                    if per == 0:
                        break
                    chUserDefinedName = chUserDefinedName + chr(per)
                print("device user define name: %s" % chUserDefinedName)

                chModelName = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                    if 0 == per:
                        break
                    chModelName = chModelName + chr(per)
                print("device model name: %s" % chModelName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                print("user serial number: %s" % strSerialNumber)
                devList.append("[" + str(i) + "]USB: " + chUserDefinedName + " " + chModelName
                               + "(" + str(strSerialNumber) + ")")

        # ui.ComboDevices.clear()
        # ui.ComboDevices.addItems(devList)
        # ui.ComboDevices.setCurrentIndex(0)

    # ch:打开相机 | en:open device
    def open_device(self):
       
        if self.isOpen:
            # QMessageBox.warning(mainWindow, "Error", 'Camera is Running!', QMessageBox.Ok)
            return MV_E_CALLORDER

        # nSelCamIndex = ui.ComboDevices.currentIndex()
        if self.nSelCamIndex < 0:
            # QMessageBox.warning(mainWindow, "Error", 'Please select a camera!', QMessageBox.Ok)
            return MV_E_CALLORDER

        self.obj_cam_operation = CameraOperation(self.cam, self.deviceList, self.nSelCamIndex)
        ret = self.obj_cam_operation.Open_device()
        if 0 != ret:
            strError = "Open device failed ret:" + ToHexStr(ret)
            # QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
            self.isOpen = False
        else:
            self.set_continue_mode()

            self.get_param()

            self.isOpen = True
            # self.enable_controls()

    # ch:开始取流 | en:Start grab image
    def start_grabbing(self,task):
      
        ret = self.obj_cam_operation.Start_grabbing(self.nSelCamIndex+1,task)
        if ret != 0:
            strError = "Start grabbing failed ret:" + ToHexStr(ret)
            # QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        else:
            self.isGrabbing = True
            # enable_controls()

    # ch:停止取流 | en:Stop grab image
    def stop_grabbing(self):

        ret = self.obj_cam_operation.Stop_grabbing()
        if ret != 0:
            strError = "Stop grabbing failed ret:" + ToHexStr(ret)
            # QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        else:
            self.isGrabbing = False
            # enable_controls()

    # ch:关闭设备 | Close device
    def close_device(self):
        
        if self.isOpen:
            self.obj_cam_operation.Close_device()
            self.isOpen = False

        self.isGrabbing = False

        # enable_controls()

    # ch:设置触发模式 | en:set trigger mode
    def set_continue_mode(self):
        strError = None

        ret = self.obj_cam_operation.Set_trigger_mode(False)
        if ret != 0:
            strError = "Set continue mode failed ret:" + ToHexStr(ret) + " mode is " 
        #     QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        # else:
        #     ui.radioContinueMode.setChecked(True)
        #     ui.radioTriggerMode.setChecked(False)
        #     ui.bnSoftwareTrigger.setEnabled(False)

    # ch:设置软触发模式 | en:set software trigger mode
    def set_software_trigger_mode(self):

        ret = self.obj_cam_operation.Set_trigger_mode(True)
        if ret != 0:
            strError = "Set trigger mode failed ret:" + ToHexStr(ret)
        #     QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        # else:
        #     ui.radioContinueMode.setChecked(False)
        #     ui.radioTriggerMode.setChecked(True)
        #     ui.bnSoftwareTrigger.setEnabled(self.isGrabbing)

    # ch:设置触发命令 | en:set trigger software
    def trigger_once(self):
        ret = self.obj_cam_operation.Trigger_once()
        if ret != 0:
            strError = "TriggerSoftware failed ret:" + ToHexStr(ret)
            # QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)

    # ch:存图 | en:save image
    def save_bmp(self):
        ret = self.obj_cam_operation.Save_Bmp()
        if ret != MV_OK:
            strError = "Save BMP failed ret:" + ToHexStr(ret)
            # QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        else:
            print("Save image success")

    # ch: 获取参数 | en:get param
    def get_param(self):
        ret = self.obj_cam_operation.Get_parameter()
        if ret != MV_OK:
            strError = "Get param failed ret:" + ToHexStr(ret)
            # QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        # else:
        #     ui.edtExposureTime.setText("{0:.2f}".format(obj_cam_operation.exposure_time))
        #     ui.edtGain.setText("{0:.2f}".format(obj_cam_operation.gain))
        #     ui.edtFrameRate.setText("{0:.2f}".format(obj_cam_operation.frame_rate))

    # ch: 设置参数 | en:set param
    def set_param(self,frame_rate,exposure,gain):
        # frame_rate = ui.edtFrameRate.text()
        # exposure = ui.edtExposureTime.text()
        # gain = ui.edtGain.text()
        ret = self.obj_cam_operation.Set_parameter(frame_rate, exposure, gain)
        if ret != MV_OK:
            strError = "Set param failed ret:" + ToHexStr(ret)
            # QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)

        return MV_OK

    # ch: 设置控件状态 | en:set enable status
    # def enable_controls():
    #     global self.isGrabbing
    #     global self.isOpen

    #     # 先设置group的状态，再单独设置各控件状态
    #     ui.groupGrab.setEnabled(self.isOpen)
    #     ui.groupParam.setEnabled(self.isOpen)

    #     ui.bnOpen.setEnabled(not self.isOpen)
    #     ui.bnClose.setEnabled(self.isOpen)

    #     ui.bnStart.setEnabled(self.isOpen and (not self.isGrabbing))
    #     ui.bnStop.setEnabled(self.isOpen and self.isGrabbing)
    #     ui.bnSoftwareTrigger.setEnabled(self.isGrabbing and ui.radioTriggerMode.isChecked())

    #     ui.bnSaveImage.setEnabled(self.isOpen and self.isGrabbing)

    # # ch: 初始化app, 绑定控件与函数 | en: Init app, bind ui and api
    # app = QApplication(sys.argv)
    # mainWindow = QMainWindow()
    # ui = Ui_MainWindow()
    # ui.setupUi(mainWindow)
    # ui.bnEnum.clicked.connect(enum_devices)
    # ui.bnOpen.clicked.connect(open_device)
    # ui.bnClose.clicked.connect(close_device)
    # ui.bnStart.clicked.connect(start_grabbing)
    # ui.bnStop.clicked.connect(stop_grabbing)

    # ui.bnSoftwareTrigger.clicked.connect(trigger_once)
    # ui.radioTriggerMode.clicked.connect(set_software_trigger_mode)
    # ui.radioContinueMode.clicked.connect(set_continue_mode)

    # ui.bnGetParam.clicked.connect(get_param)
    # ui.bnSetParam.clicked.connect(set_param)

    # ui.bnSaveImage.clicked.connect(save_bmp)

    # mainWindow.show()

    # app.exec_()

    # close_device()

    # sys.exit()
