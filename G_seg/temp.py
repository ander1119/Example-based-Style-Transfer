import numpy as np 
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("input_pic/6.png", cv2.IMREAD_GRAYSCALE)

col_arr = np.arange(image.shape[0])
col_arr = np.transpose(np.tile(col_arr, (image.shape[1], 1)))

row_arr = np.arange(image.shape[1]).reshape(1, -1)
row_arr = np.tile(row_arr, (image.shape[0], 1))

plt.imshow(np.stack([col_arr, row_arr, np.zeros_like(col_arr)], axis=2))
plt.show()

