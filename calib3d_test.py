from camera_calibrator import CameraCalibrator

if __name__ == "__main__":
    # Test 3d calibration
    calib = CameraCalibrator(board_shape=(6, 7), tile_side=0.062, apriltag_families="tag36h10")
    image = cv2.imread("data/test2/amplitude3.png", cv2.IMREAD_GRAYSCALE)
    xyz_coordinates_matrix = np.load("data/test2/xyz.npy")
    print(xyz_coordinates_matrix.shape)
    print(type(xyz_coordinates_matrix[0][0][0]))
    M_2d = calib.get_extrinsics_2D(image)
    # calib.plot()
    # Fix order
    xyz_coordinates_matrix_ordered = xyz_coordinates_matrix
    xyz_coordinates_matrix_ordered[:, :, 0] = xyz_coordinates_matrix[:, :, 1]
    xyz_coordinates_matrix_ordered[:, :, 1] = xyz_coordinates_matrix[:, :, 2]
    xyz_coordinates_matrix_ordered[:, :, 2] = xyz_coordinates_matrix[:, :, 0]

    # cv2.imshow("X", xyz_coordinates_matrix[:, :, 0])
    # cv2.waitKey(0)

    # cv2.imshow("Y", xyz_coordinates_matrix[:, :, 1])
    # cv2.waitKey(0)

    # cv2.imshow("Z", xyz_coordinates_matrix[:, :, 2])
    # cv2.waitKey(0)

    M_3d = calib.get_extrinsics_3D(image, xyz_coordinates_matrix_ordered)
    print(M_2d)
    print("·")
    print(M_3d)