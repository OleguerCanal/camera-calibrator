from camera_calibrator import CameraCalibrator
import numpy as np
import cv2
import pickle
import copy
from glob import glob

def exact_test():
    A1 = [[-0.989992, -0.141120, 0.000000, 0],
        [0.141120, -0.989992, 0.000000, 0],
        [0.000000, 0.000000, 1.000000, 0],
        [0, 0, 0, 1]]
    B1 = [[-0.989992, -0.138307, 0.028036, -26.9559],
        [0.138307, -0.911449, 0.387470, -96.1332],
        [-0.028036, 0.387470, 0.921456, 19.4872],
        [0, 0, 0, 1]]
    A2 = [[0.070737, 0.000000, 0.997495, -400.000],
        [0.000000, 1.000000, 0.000000, 0.000000],
        [-0.997495, 0.000000, 0.070737, 400.000],
        [0, 0, 0, 1]]
    B2 = [[0.070737, 0.198172, 0.997612, -309.543],
        [-0.198172, 0.963323, -0.180936, 59.0244],
        [-0.977612, -0.180936, 0.107415, 291.177],
        [0, 0, 0, 1]]

    As = [A1, A2]
    Bs = [B1, B2]
    calibrator = CameraCalibrator()
    x = calibrator.solve_axxb_exact(zip(As, Bs))
    # x = calibrator.solve_axxb(zip(As, Bs))
    print(x)
    
if __name__ == "__main__":
    #exact_test()
    calib = CameraCalibrator(board_shape=(6, 7), tile_side=0.010, apriltag_families="tag36h10")
    
    Ta_is = []
    Tb_is = []

    paths = glob("data/157*/")

    for path in paths:
        print(path)
        image = cv2.imread(path + "amplitude.png", cv2.IMREAD_GRAYSCALE)
        A_trans = np.load(path + "translation.npy")
        A_rot = np.load(path + "rotation.npy")
        f = open(path + "/xyz.pkl", "rb")
        xyz_coordinates_matrix = pickle.load(f)
        
        # Fix order
        xyz_coordinates_matrix_ordered = copy.deepcopy(xyz_coordinates_matrix)
        xyz_coordinates_matrix_ordered[:, :, 0] = xyz_coordinates_matrix[:, :, 1]
        xyz_coordinates_matrix_ordered[:, :, 1] = xyz_coordinates_matrix[:, :, 2]
        xyz_coordinates_matrix_ordered[:, :, 2] = xyz_coordinates_matrix[:, :, 0]

        # Compute Tb_i
        # A = calib.chessboard_extrinsics_2D(image)
        Tb_i = calib.chessboard_extrinsics_3D(image, xyz_coordinates_matrix_ordered)
        Tb_is.append(Tb_i)
        Ta_i = calib.transquat_to_mat(A_trans, A_rot)
        print(Ta_i)
        Ta_is.append(Ta_i)
        calib.plot()
        # print(A)
        print(Tb_i)

    X = calib.eye_in_hand_finetunning(Ta_is, Tb_is)
    print("X:")
    print(X)
    print("Before:")
    print(Ta_is[0])
    print("After:")
    print(np.dot(Ta_is[0],X))