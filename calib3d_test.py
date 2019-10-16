from camera_calibrator import CameraCalibrator
import numpy as np
import cv2
import pickle
import copy

if __name__ == "__main__":
    # Test 3d calibration
    calib = CameraCalibrator(board_shape=(6, 7), tile_side=0.062, apriltag_families="tag36h10")
    image = cv2.imread("data/test5/amplitude.png", cv2.IMREAD_GRAYSCALE)
    # xyz_coordinates_matrix = np.load("data/test4/xyz.npy", allow_pickle=True)
    f = open("data/test5/xyz.pkl", "rb")
    # print(f)
    # with open("data/test4/xyz.npy", 'rb') as f:
    #     xyz_coordinates_matrix = pickle.load(f)
    xyz_coordinates_matrix = pickle.load(f)
    
    # print(xyz_coordinates_matrix.shape)
    # print(type(xyz_coordinates_matrix[0][0][0]))
    # # M_2d = calib.chessboard_extrinsics_2D(image)
    # # calib.plot()
    # # Fix order
    xyz_coordinates_matrix_ordered = copy.deepcopy(xyz_coordinates_matrix)
    xyz_coordinates_matrix_ordered[:, :, 0] = xyz_coordinates_matrix[:, :, 1]
    xyz_coordinates_matrix_ordered[:, :, 1] = xyz_coordinates_matrix[:, :, 2]
    xyz_coordinates_matrix_ordered[:, :, 2] = xyz_coordinates_matrix[:, :, 0]

    # print(xyz_coordinates_matrix_ordered[0, 0, 0])
    # print(xyz_coordinates_matrix_ordered[100, 0, 0])

    # for i in range(xyz_coordinates_matrix_ordered.shape[0]):
    #     print(xyz_coordinates_matrix_ordered[50, i, 0])

    # for j in range(xyz_coordinates_matrix_ordered.shape[0]):
    #     print(xyz_coordinates_matrix_ordered[50, j, 2])

    # print(xyz_coordinates_matrix_ordered[0, 0, 0])
    # print(xyz_coordinates_matrix_ordered[0, 200, 0])

    cv2.imshow("X", xyz_coordinates_matrix_ordered[:, :, 0])
    cv2.waitKey(0)

    cv2.imshow("Y", xyz_coordinates_matrix_ordered[:, :, 1])
    cv2.waitKey(0)

    cv2.imshow("Z", xyz_coordinates_matrix_ordered[:, :, 2])
    cv2.waitKey(0)

    M_3d = calib.chessboard_extrinsics_3D(image, xyz_coordinates_matrix_ordered)
    print(M_2d)
    print(".")
    print(M_3d)