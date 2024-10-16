'''
A simple Program for grabing video from basler camera and converting it to opencv img.
Tested on Basler acA1920-25uc (USB3, Window x64 , python 3.12)

'''
from pypylon import pylon
import cv2
import time
import numpy as np
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.Width.Value = 1722
camera.Height.Value = 960
camera.OffsetX.Value = 0
camera.OffsetY.Value = 0
camera.ExposureTime.SetValue(10000)
camera.Gain.SetValue(20)
camera.AcquisitionFrameRate.SetValue(20)

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()

converter.OutputPixelFormat = pylon.PixelType_RGB16packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_LsbAligned

while camera.IsGrabbing():
    startTime = time.time()
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        image = grabResult.GetArray()
        cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Video', image)
        k = cv2.waitKey(1)
        if k == 27:
            break
    grabResult.Release()

    runningTime = (time.time() - startTime)
    fps = 1.0/runningTime
    print ("%f  FPS" % fps)

# Releasing the resource    
camera.StopGrabbing()
cv2.destroyAllWindows()