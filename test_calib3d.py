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
    
    Ta_is = []
    Tb_is = []

    paths = glob("data/157*/")

    for path in paths:
        image = cv2.imread(path + "amplitude.png", cv2.IMREAD_GRAYSCALE)
        A_trans = np.load(path + "translation.npy")
        A_rot = np.load(path + "rotation.npy")
        f = open(path + "/xyz.pkl", "rb")
        xyz_coordinates_matrix = pickle.load(f)

        world_to_cam = np.linalg.inv(calib.transquat_to_mat(A_trans, A_rot))
        cam_to_chess = calib.chessboard_extrinsics_3D(image, xyz_coordinates_matrix)
        Ta_is.append(np.mat(world_to_cam))
        Tb_is.append(np.mat(cam_to_chess))

        # Debug
        print("world_to_cam:")
        print(world_to_cam)

        world_to_chess = np.dot(cam_to_chess, world_to_cam)
        print("World to chess:")
        pmat(world_to_chess)
        print(np.dot(world_to_chess, np.array([1, 1, 1, 1])))
        


    X = calib.eye_in_hand_finetunning(Ta_is, Tb_is)
    print("X:")
    pmat(X)
    # print("Before:")
    # pmat(Ta_is[0])
    # print("After:")
    # pmat(np.dot(X, Ta_is[0]))
