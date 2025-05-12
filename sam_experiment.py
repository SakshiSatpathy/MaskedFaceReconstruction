import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import os

"""
This code allows the user to segment the original images using SAM.
Note: you have to manually update the filepaths within the code.
Running the code will prompt a small GUI to select points to include/exclude in the segmented image.
Left-clicking on a point (shown as a blue dot) will INCLUDE that point, and Right-Clicking (shown as a red dot) will EXCLUDE that point.
"""



# Load SAM Model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)


image = cv2.imread("filename.jpg") #NOTE: Replace image path here with image path.
height, width, channels = image.shape
img = image.copy()

numClicks = 0
done_selecting = False
input_points = []
input_labels = []

def drawfunction(event, x, y, flags, param):
    global drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        input_points.append([x, y])
        input_labels.append(1)
        cv2.circle(img, (x, y), 10, (255, 0, 0), -1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        input_points.append([x, y])
        input_labels.append(0)
        cv2.circle(img, (x, y), 10, (0, 0, 255), -1)

# Allows the user to select points on what the segmented image should include (Left Click) and Exclude (Right Click)
cv2.namedWindow('image')
cv2.setMouseCallback('image', drawfunction)
while(1):
    cv2.imshow('image', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


predictor.set_image(image)
input_points = np.array(input_points)
input_labels = np.array(input_labels)

masks, _, _ = predictor.predict(
    point_coords = input_points,
    point_labels = input_labels,
    multimask_output=False
)


color = np.array([30/255, 144/255, 255/255, 0.6])
h, w = masks.shape[-2:] 
mask_image = masks.reshape(h, w, 1) * color.reshape(1, 1, -1)

mask_image = (mask_image*255).astype(np.uint8)

#Creating segmented image with correct colors:
blank_img = np.zeros((height, width, 3), np.uint8)

for y in range(height):
    for x in range(width):
        if mask_image[y, x, 0] != 0:
            blank_img[y, x, 0] = image[y, x, 0]
            blank_img[y, x, 1] = image[y, x, 1]
            blank_img[y, x, 2] = image[y, x, 2]
cv2.imwrite("output_filepath_segmented.jpg", blank_img) #NOTE: replace with output filepath.