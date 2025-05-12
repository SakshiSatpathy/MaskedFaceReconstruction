import numpy as np 
import cv2 
from pynput.mouse import Listener


"""
This file is used for preprocessing before running images through EdgeConnect.
When run, the code prompts the user to draw on an image, creating a white binary mask.
As required by EdgeConnect, this generates a white binary mask, as well as an image with the binary mask overlayed.
This binary mask represents the area that must be infilled by EdgeConnect.
Note: the filenames and filepaths of the input and output images can be manually changed within the code.
"""

def drawfunction(event, x, y, flags, param):
    global drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img,(x,y),10,(255,255,255),-1)
            cv2.circle(blank_img, (x,y),10, (255,255,255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


img = cv2.imread("filename.jpg") # NOTE: filename and path of input image must be changed.
img_copy = np.copy(img)
gray_img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_img_copy, threshold1=100, threshold2=200)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.namedWindow('image')
cv2.setMouseCallback('image', drawfunction)

global blank_img
blank_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)



drawing = False
while(1):
    cv2.imshow('image', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.imwrite("img_and_bm.jpg", img) #NOTE: filename and path of outputs must be manually changed.
cv2.imwrite("binary_mask.jpg", blank_img)
cv2.imwrite("edge_map.jpg", edges)

cv2.destroyAllWindows()