import numpy as np
import cv2

a = np.load("color.npy")
cv2.imshow("a", a)
cv2.waitKey(0)