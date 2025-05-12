import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from torchvision.models import resnet18
#import tkinter as tk
#from tkinter.filedialog import askopenfilename
import os


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# Preprocessing function
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0), image


#Tk().withdraw()

#filename = askopenfilename()
#base, ext = os.path.splitext(filename)

# Load SAM Model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)


image = cv2.imread("mask_test_14.jpg")
height, width, channels = image.shape
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        #add stars to the image
        cv2.circle(img, (x, y), 10, (255, 0, 0), -1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        input_points.append([x, y])
        input_labels.append(0)
        cv2.circle(img, (x, y), 10, (0, 0, 255), -1)

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
mask_image = masks.reshape(h, w, 1) * color.reshape(1, 1, -1) # for some reason this mask is getting NO data in it.
plt.gca().imshow(mask_image)
plt.axis('off')
plt.show() #has the correct data

mask_image = (mask_image*255).astype(np.uint8)
 #Currently, however, we are just painting the segment itself; We instead want to paint the color of the pixels for each pixel in the segment.

#Making the image with correct colors:
blank_img = np.zeros((height, width, 3), np.uint8)

for y in range(height):
    for x in range(width):
        if mask_image[y, x, 0] != 0:
            blank_img[y, x, 0] = image[y, x, 0]
            blank_img[y, x, 1] = image[y, x, 1]
            blank_img[y, x, 2] = image[y, x, 2]
#blank_img = cv2.cvtColor(blank_img, cv2.COLOR_RGB2BGR)
cv2.imwrite(f"3DMM-Fitting-Pytorch/sam_output/mask_test_14_segmented.jpg", blank_img)


"""

TESTING whether the image output is working.

image_output = cv2.imread('mask.png')
image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,10))
plt.imshow(image_output)
plt.axis('on')
plt.show() 

"""


"""
# Feature Extraction using ResNet-18
resnet = resnet18(pretrained=True)
resnet.eval()

def extract_features(image_tensor):
    with torch.no_grad():
        features = resnet(image_tensor)
    return features

# Depth Estimation Placeholder
def estimate_depth(image_tensor):
    # Placeholder for a depth estimation model
    depth_map = torch.rand((1, 1, 224, 224))  # Random tensor simulating depth map
    return depth_map

# Placeholder for 3D Face Modeling using 3DMM or Gaussian Splatting
def reconstruct_3d_face(features, depth_map):
    # Simulate 3D reconstruction using dummy output
    return np.random.rand(3, 224, 224)  # Random 3D face representation

# Full pipeline
def process_face(image_path):
    image_tensor, original_image = preprocess_image(image_path)
    features = extract_features(image_tensor)
    depth_map = estimate_depth(image_tensor)
    reconstructed_face = reconstruct_3d_face(features, depth_map)
    return original_image, reconstructed_face

# Display multiple images
def display_results(image_paths):
    fig, axes = plt.subplots(len(image_paths), 2, figsize=(10, len(image_paths) * 5))
    if len(image_paths) == 1:
        axes = [axes]

    for i, image_path in enumerate(image_paths):
        original_image, reconstructed_face = process_face(image_path)

        # Display original image
        axes[i][0].imshow(original_image)
        axes[i][0].set_title("Original Image")
        axes[i][0].axis("off")

        # Display reconstructed 3D face as a depth map projection
        axes[i][1].imshow(reconstructed_face[0], cmap='gray')
        axes[i][1].set_title("Reconstructed 3D Face")
        axes[i][1].axis("off")

    plt.show()"""