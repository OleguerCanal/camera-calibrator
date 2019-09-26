import apriltag
import cv2
import numpy as np

def get_corners(img, boardSize, subpixel = False):
    # Convert to grayscale if its not
    if (len(img.shape) != 2):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    found, corners = cv2.findChessboardCorners(img, boardSize)

    if not found or not subpixel:
        return found, corners

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # termination criteria
    corners_subpx = cv2.cornerSubPix(img, corners, boardSize, (-1,-1), criteria)  # subpixel accuracy

    # TODO(Oleguer): Detect origin using apriltag and reorient
    return True, corners_subpx

def get_apriltag_center(img):
    detector = apriltag.Detector()
    result = detector.detect(img)
    return np.array(result[0].center, dtype=np.int16)

def sorted_corners(img, boardSize):
    found, corners = get_corners(img, boardSize)
    apriltag_center = get_apriltag_center(img)
    return found, corners, apriltag_center

if __name__ == "__main__":
    # gray_img = cv2.imread("data/calib_example.png", cv2.IMREAD_GRAYSCALE)
    # apriltag_img = cv2.imread("data/apriltag_example.jpeg", cv2.IMREAD_GRAYSCALE)
    img = cv2.imread("data/collage.png", cv2.IMREAD_GRAYSCALE)
    boardSize = (4, 4)

    found, corners, apriltag_center = sorted_corners(img, boardSize)
    print(corners.shape)
    # print(corners)
    corners = np.reshape(corners, (boardSize[0], boardSize[1], 1, 2))
    print(corners.shape)
    print(corners)
    # Corners stored in a matrix way
    corners = np.rot90(corners)
    print(corners.shape)
    print(corners)




    # Show
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # cv2.circle(img, (apriltag_center[0], apriltag_center[1]), 5, (0, 0, 255), -1)
    # img = cv2.drawChessboardCorners(img, boardSize, corners, found)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)