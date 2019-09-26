import apriltag
import cv2
import numpy as np

def get_corners(img, boardSize, subpixel = False):
    # Convert to grayscale if its not
    if (len(img.shape) != 2):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    found, corners = cv2.findChessboardCorners(img, boardSize)

    if not found or not subpixel:
        return False, corners

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # termination criteria
    corners_subpx = cv2.cornerSubPix(img, corners, boardSize, (-1,-1), criteria)  # subpixel accuracy

    # TODO(Oleguer): Detect origin using apriltag and reorient
    return True, corners_subpx

def get_apriltag_center(img):
    detector = apriltag.Detector()
    result = detector.detect(img)
    return np.array(result[0].center, dtype=np.int16)

if __name__ == "__main__":
    # gray_img = cv2.imread("data/calib_example.png", cv2.IMREAD_GRAYSCALE)
    # boardSize = (4, 4)

    apriltag_img = cv2.imread("data/apriltag_example.jpeg", cv2.IMREAD_GRAYSCALE)
    apriltag_center = get_apriltag_center(apriltag_img)
    print(apriltag_center)

    apriltag_rgb_img = cv2.cvtColor(apriltag_img, cv2.COLOR_GRAY2RGB)
    cv2.circle(apriltag_rgb_img, (apriltag_center[0], apriltag_center[1]), 3, (0, 0, 255), -1)
    cv2.imshow("apriltag_rgb_img", apriltag_rgb_img)
    cv2.waitKey(0)


    # found, corners = get_corners(gray_img, boardSize, False)

    # # Show
    # rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    # rgb_img = cv2.drawChessboardCorners(rgb_img, boardSize, corners, found)
    # cv2.imshow("rgb_img", rgb_img)
    # cv2.waitKey(0)