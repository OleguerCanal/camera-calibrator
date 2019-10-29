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
        # pmat(calib.transquat_to_mat(A_trans, A_rot))

        world_to_cam = np.linalg.inv(calib.transquat_to_mat(A_trans, A_rot))
        reordered = copy.deepcopy(xyz_coordinates_matrix)
        reordered[:, :, 0] = -xyz_coordinates_matrix[:, :, 1]
        reordered[:, :, 1] = -xyz_coordinates_matrix[:, :, 2]
        reordered[:, :, 2] = xyz_coordinates_matrix[:, :, 0]
        cam_to_chess = calib.chessboard_extrinsics_3D(image, reordered)
        Ta_is.append(np.mat(world_to_cam))
        Tb_is.append(np.mat(cam_to_chess))

        # Debug
        print("world_to_cam:")
        pmat(world_to_cam)
        pmat(np.dot(world_to_cam, np.array([0, 0, 0, 1])))

        print("cam_to_chess:")
        pmat(cam_to_chess)

        world_to_chess = np.dot(cam_to_chess, world_to_cam)
        # world_to_chess = np.dot(world_to_cam, cam_to_chess)
        print("World to chess:")
        pmat(np.linalg.inv(world_to_chess))
        # pmat(np.dot(world_to_chess, np.array([1, 1, 1, 1])))

        calib.plot()


    X = calib.eye_in_hand_finetunning(Ta_is, Tb_is)
    print("X:")
    pmat(X)

    print("world to chess")
    for world_to_cam, cam_to_chess in zip(Ta_is, Tb_is):
        pmat(np.linalg.inv(cam_to_chess*X*world_to_cam))
        pmat(np.linalg.inv(cam_to_chess*np.linalg.inv(X)*world_to_cam))
