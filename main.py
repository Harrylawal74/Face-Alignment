import numpy as np
import cv2
import matplotlib.pyplot as plt

image = 'face.png'


height, width = image.shape[:2]
num_rows, num_cols = 4, 4

x_coords = np.linspace(0, width - 1, num=num_cols, dtype=int)
y_coords = np.linspace(0, height - 1, num=num_rows, dtype=int)

grid_points = [(x, y) for y in y_coords for x in x_coords]


keypoints = [cv2.KeyPoint(x=float(x), y=float(y), _size=16) for (x, y) in grid_points]

sift = cv2.SIFT_create()
keypoints, descriptors = sift.compute(image, keypoints)
