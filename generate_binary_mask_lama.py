import numpy as np 
import cv2 
from pynput.mouse import Listener


"""
This file is used for preprocessing before running images through LAMA.
When run, the code prompts the user to draw on an image, creating a BLACK binary mask.
This binary mask represents the area that must be infilled by LAMA.
Note: the filenames and filepaths of the input and output images can be manually changed within the code.
"""

def drawfunction(event, x, y, flags, param):
    global drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img,(x,y),10,(0,0,0),-1)
            cv2.circle(blank_img, (x,y),10, (0,0,0), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


img = cv2.imread("filename.jpg") # NOTE: filename and path of input image must be changed.
img_copy = np.copy(img)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.namedWindow('image')
cv2.setMouseCallback('image', drawfunction)

global blank_img
blank_img = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8)
blank_img = blank_img * 255


drawing = False
while(1):
    cv2.imshow('image', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

 #NOTE: filename and path of outputs must be manually changed.
cv2.imwrite("image1_mask001.jpg", blank_img)

cv2.destroyAllWindows()