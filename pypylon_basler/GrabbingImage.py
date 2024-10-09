from pypylon import pylon
import cv2
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# demonstrate some feature access
new_width = camera.Width.Value - camera.Width.Inc
if new_width >= camera.Width.Min:
    camera.Width.Value = new_width

numberOfImagesToGrab = 100
camera.StartGrabbingMax(numberOfImagesToGrab)

camera.IsGrabbing()
grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

if grabResult.GrabSucceeded():
    # Access the image data.
    print("SizeX: ", grabResult.Width)
    print("SizeY: ", grabResult.Height)
    img = grabResult.Array
    print("Gray value of first pixel: ", img[0, 0])
    cv2.imshow('ccc',img)
    cv2.imwrite(r'C:\Users\CCSX009\Documents\yolov5\test_image\camera1\a.jpg',img)
# grabResult.Release()
# camera.Close()

from pypylon import pylon

# Initialize camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Set Gain Auto parameters
minLowerLimit = camera.AutoGainRawLowerLimit.Min
maxUpperLimit = camera.AutoGainRawUpperLimit.Max
camera.AutoGainRawLowerLimit.Value = minLowerLimit
camera.AutoGainRawUpperLimit.Value = maxUpperLimit
camera.AutoTargetValue.Value = 150

# Set Exposure Auto parameters
minLowerLimit = camera.AutoExposureTimeAbsLowerLimit.Min
maxUpperLimit = camera.AutoExposureTimeAbsUpperLimit.Max
camera.AutoExposureTimeAbsLowerLimit.Value = minLowerLimit
camera.AutoExposureTimeAbsUpperLimit.Value = maxUpperLimit
camera.AutoTargetBrightness.Value = 0.6

# Enable Gain and Exposure Auto
camera.GainAuto.Value = "Continuous"
camera.ExposureAuto.Value = "Continuous"

# Start grabbing
camera.StartGrabbing()