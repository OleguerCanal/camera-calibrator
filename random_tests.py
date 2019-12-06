from camera_calibrator import CameraCalibrator
import numpy as np
import cv2
import pickle
import copy
from glob import glob

def pmat(mat):
    '''Print matrix
    '''
    print(np.round(mat, 3))

if __name__ == "__main__":
    calib = CameraCalibrator(board_shape=(6, 7), tile_side=0.10, apriltag_families="tag36h10")
    image = cv2.imread("data/amplitude.png", cv2.IMREAD_GRAYSCALE)
    a = calib.get_apriltag_center(image)
    print(a)