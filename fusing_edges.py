import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

angle_0 = cv2.imread("angled_img_0.jpg", cv2.IMREAD_GRAYSCALE)
angle_1 = cv2.imread("angled_img_1.jpg", cv2.IMREAD_GRAYSCALE)
angle_2 = cv2.imread("angled_img_2.jpg", cv2.IMREAD_GRAYSCALE)
angle_3 = cv2.imread("angled_img_3.jpg", cv2.IMREAD_GRAYSCALE)
angle_4 = cv2.imread("angled_img_4.jpg", cv2.IMREAD_GRAYSCALE)

edges_0 = cv2.Canny(angle_0,100,200)
edges_1 = cv2.Canny(angle_1,100,200)
edges_2 = cv2.Canny(angle_2,100,200)
edges_3 = cv2.Canny(angle_3,100,200)
edges_4 = cv2.Canny(angle_4,100,200)

edge_list = [edges_0, edges_1, edges_2, edges_3, edges_4]

fused = np.mean(edge_list, axis=0).astype(np.uint8)

for i in range(5):
    cv2.imwrite(f"angle_{i}_edges.jpg", edge_list[i])
cv2.imwrite("fused_img.jpg", fused)