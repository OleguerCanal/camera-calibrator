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
    print("")

def inv(mat):
    return np.linalg.inv(mat)

def construct_xyz_coordinate_matrix(depth, intrinsics):
    inv = np.linalg.inv(intrinsics)
    def px_to_m(i, j):
        cen = (i, j, 1)
        point_m = np.dot(inv, cen)
        return point_m[0], point_m[1], depth[i][j]

    xyz_coord = np.empty((depth.shape[0], depth.shape[1], 3), dtype = np.float32)
    print(xyz_coord.shape[0])
    print(xyz_coord.shape[1])
    for i in range(xyz_coord.shape[0]):
        for j in range(xyz_coord.shape[1]):
            x, y, z  = px_to_m(i, j)
            xyz_coord[i][j][0] = y
            xyz_coord[i][j][1] = x
            xyz_coord[i][j][2] = z
    return xyz_coord

if __name__ == "__main__":
    calib = CameraCalibrator(board_shape=(6, 7), tile_side=0.10, apriltag_families="tag36h10")
    Ta_is = []
    Tb_is = []

    paths = glob("data/157*/")

    xyz_coordinates_matrix = None
    for path in paths:
        if path == "data/1574786828/":
            continue
        depth = np.load(path + "depth.npy")
        color = np.load(path + "color.npy")
        intrinsics = np.load(path + "intrinsics.npy")
        A_trans = np.load(path + "translation.npy")
        A_rot = np.load(path + "rotation.npy")

        mount_to_world = calib.transquat_to_mat(A_trans, A_rot)  # point_world = mount_to_world * point_mount
        world_to_mount = inv(mount_to_world)

        if xyz_coordinates_matrix is None:
            xyz_coordinates_matrix = construct_xyz_coordinate_matrix(depth, intrinsics) 

        image = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        # optical_to_chess = calib.chessboard_extrinsics_3D(image, xyz_coordinates_matrix)
        optical_to_chess = calib.chessboard_extrinsics_2D(image)
        # Ta_is.append(np.mat(world_to_cam))
        # Tb_is.append(np.mat(cam_to_chess))

        mount_to_optical = np.matrix(np.array([[0, 0, 1, 0],
                                        [-1, 0, 0, 0],
                                        [0, -1, 0, 0],
                                        [0, 0, 0, 1]]))

        # print("world_to_mount:")
        # pmat(world_to_mount)

        # print("mount_to_optical:")
        # pmat(mount_to_optical)

        print("optical_to_chess:")
        pmat(optical_to_chess)

        # print("Closed loop:")
        # chess_to_world = inv(world_to_mount)*mount_to_optical*optical_to_chess
        # pmat(chess_to_world)


    # X = calib.eye_in_hand_finetunning(Ta_is, Tb_is)
    # print("X:")
    # pmat(X)

    # print("world to chess")
    # for world_to_cam, cam_to_chess in zip(Ta_is, Tb_is):
    #     pmat(np.linalg.inv(cam_to_chess*X*world_to_cam))
    #     pmat(np.linalg.inv(cam_to_chess*np.linalg.inv(X)*world_to_cam))
