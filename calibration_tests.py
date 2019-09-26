import numpy as np
import cv2

if __name__ == "__main__":
    img = cv2.imread("calib_example.png", cv2.IMREAD_GRAYSCALE)
    boardSize = (4, 4)
    found, corners = cv2.findChessboardCorners(img, boardSize)
    print(found)
    print(corners)

    rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    rgb_img = cv2.drawChessboardCorners(rgb_img, boardSize, corners, found)
    cv2.imshow("rgb_img", rgb_img)
    cv2.waitKey(0)
    