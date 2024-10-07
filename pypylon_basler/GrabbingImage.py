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