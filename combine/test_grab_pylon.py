import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent.parent
ultralytics_main_dir = current_dir
sys.path.append(str(ultralytics_main_dir))
from base.IOConnection.basler_pylon.PylonExportImgBuffer import *
import queue
import cv2
request = Basler_Pylon_xFunc(25039617,'UserSet1')
task= queue.Queue()
img_buffer = []
request.Start_grabbing(task)
img_buffer.append(task.get())
cv2.imwrite('ssssaaaaa.jpg',img_buffer[0])
print(img_buffer[0])
# request.Stop_grabbing()