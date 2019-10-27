import apriltag
import cv2
import numpy as np
from scipy.spatial import distance as dist
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm, sqrtm, inv, expm
from scipy.optimize import minimize
from math import sqrt
from utils.rigid_trans_test import rigid_transform_3D  #TODO(oleguer): Remove that
import copy

class CameraCalibrator():
    def __init__(self, board_shape = (3, 4), tile_side = 0.062, apriltag_families = "tag36h10"):
        self.board_shape = board_shape
        self.tile_side = tile_side
        self.apriltag_families = apriltag_families

    def transquat_to_mat(self, translation, quaternion):
        ''' Given a translation vector and a quaternion returns transformation matrix
        '''
        M = np.eye(4)
        rot_matrix = R.from_quat(quaternion).as_dcm()
        M[0:3,0:3] = rot_matrix
        M[0:3, 3] = translation
        return M

    #TODO(oleguer): Return intrinsics as well
    def chessboard_extrinsics_2D(self, image):
        '''Given checkerboard+apriltag 2D image, returns extrinsics:
        transformation matrix from checkerboard to camera frame
        '''
        self.corners = self.__get_oriented_corners(image)

        # 2. Get matrix of chessboard frame values
        corners_m = []
        for i in range(self.board_shape[1]): #TODO(oleguer): Review board_shape params!
            for j in range(self.board_shape[0]):
                corners_m.append([i*self.tile_side, j*self.tile_side, 0])
        corners_m = [np.array(corners_m, dtype=np.float32)]

        # 3. Compute transformation
        ret, intrinsics_mat, distortion_coef, rotation_vect, translation_vect =\
            cv2.calibrateCamera(corners_m, [self.corners], self.image.shape[::-1], None, None)
        rotation_vect = np.array(rotation_vect).reshape(3)
        translation_vect = np.array(translation_vect).reshape(3)
        rot_matrix = R.from_rotvec(rotation_vect).as_dcm()

        # From camera to checkerboard
        # camera_to_chessboard = np.eye(4)
        # camera_to_chessboard[0:3,0:3] = rot_matrix
        # camera_to_chessboard[0:3, 3] = translation_vect
        
        # From checkerboard to camera
        chessboard_to_camera = np.eye(4)
        chessboard_to_camera[0:3,0:3] = rot_matrix.T
        chessboard_to_camera[0:3, 3] = -translation_vect
        return chessboard_to_camera

    def chessboard_extrinsics_3D(self, image, xyz_coordinates_matrix):
        ''' Given image of a checkerboard with apriltag and xyz_coordinates_matrix,
            Returns transformation matrix from checkerboard to camera frame
            
            xyz_coordinates_matrix meaning: a 3 channel numpy array where:
            channel 0: x coordinates
            channel 1: y coordinates
            channel 3: z coordinates
            NOTE: Make sure the channels are in the rigth order (opencv can mess it up)!!!! 

            (Use this if you have a 3D sensor, otherwise use get_extrinsics_2D)
        '''
        # 1. Get oriented corners
        self.corners = self.__get_oriented_corners(image)
        # self.plot()

        # 2. Get matrix of camera frame values
        corners_camera_frame = []
        xyz_coordinates_matrix = np.array(xyz_coordinates_matrix)
        for corner in self.corners:
            corner = corner[0]
            x = xyz_coordinates_matrix[int(corner[1])][int(corner[0])][0]
            y = xyz_coordinates_matrix[int(corner[1])][int(corner[0])][1]
            z = xyz_coordinates_matrix[int(corner[1])][int(corner[0])][2]
            corner_camera = [x, y, z]
            corners_camera_frame.append(corner_camera)
        cb_frame_copy = copy.deepcopy(corners_camera_frame)

        # 3. Get matrix of chessboard frame values
        corners_chessboard_frame = []
        for i in range(self.board_shape[1]): #TODO(oleguer): Review board_shape params!
            for j in range(self.board_shape[0]):
                corners_chessboard_frame.append([j*self.tile_side, i*self.tile_side, 0])
        camera_frame_copy = copy.deepcopy(corners_chessboard_frame)

        camera_to_chess = self.__rigid_transform_3D(
            np.array(corners_camera_frame), np.array(corners_chessboard_frame))

        # Reproject to check if it works
        reprojected = []
        for xyz_corner in camera_frame_copy:
            repro = np.dot(np.linalg.inv(camera_to_chess), np.append(xyz_corner, [1]))
            reprojected.append(repro)
        self.__reproject(image, xyz_coordinates_matrix, reprojected)

        error = 0
        for xyz_corner, cam_frame in zip(cb_frame_copy, camera_frame_copy):
            repro = np.dot(camera_to_chess, np.append(xyz_corner, [1]))
            error += dist.euclidean(repro[0:3], cam_frame)
        error = error/len(cb_frame_copy)
        print("Reprojection error: " + str(np.round(1000*error, 2)) + " mm")

        return camera_to_chess

    def __reproject(self, image, xyz_coordinates_matrix, xyz_points):
        xyz_coordinates_matrix
        for point in xyz_points:
            # print(point)
            coord = copy.deepcopy(xyz_coordinates_matrix)
            coord[:, :, 0] -= point[0]
            coord[:, :, 1] -= point[1]
            coord[:, :, 2] -= point[2]
            dist = np.sqrt(np.square(coord[:, :, 0]) + np.square(coord[:, :, 1]) + np.square(coord[:, :, 2]))
            image_point = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
            image_point = (image_point[1], image_point[0])
            image = cv2.drawMarker(image, image_point, (255))
        
            # cv2.imshow("image", image)
            # cv2.waitKey(0)


    def eye_in_hand_finetunning(self, Ta_is, Tb_is):
        ''' 
        '''
        ABs = []
        for i in range(0, len(Ta_is)):
            for j in range(i+1, len(Ta_is)):
                A_i = np.mat(Ta_is[i])
                A_j = np.mat(Ta_is[j])
                A = inv(A_i)*A_j
                B_i = np.mat(Tb_is[i])
                B_j = np.mat(Tb_is[j])
                B = inv(B_i)*B_j  # TODO(Oleguer): Review this
                ABs.append((A, B))

        return self.solve_axxb(ABs)

    def solve_axxb(self, ABs): #TODO(oleguer): Refactor this!!



        def get_repr_vect(mat):
            return np.mat((mat[2, 1], -mat[2, 0], mat[1, 0]))
        
        # 3. Solve AX=XB problem
        M = np.mat(np.zeros((3, 3)))
        for A, B in ABs:
            A = np.mat(A)
            B = np.mat(B)
            alpha_m = logm(A[0:3, 0:3])
            alpha_v = np.mat(get_repr_vect(alpha_m))
            beta_m = logm(np.mat(B[0:3, 0:3]))
            beta_v = np.mat(get_repr_vect(beta_m))
            # print(beta_v.T)
            # print(alpha_v)
            M += beta_v.T*alpha_v
            # print(M)
        print("M: ")
        print(M)
        rot_x = sqrtm(inv(M.T*M))*M.T
        print("rot_x: ")
        print(rot_x)

        #TODO(oleguer): Take a look at this we should do least square error instead
        #OLD CODE WHICH SHOULD BE BETTER
        c_s = []
        d_s = []
        print(len(ABs))
        for A, B in ABs:
            A = np.mat(A)
            B = np.mat(B)
            c_val = np.mat(A[0:3, 0:3] - np.eye(3)) #NOTE: Swapped that for minimization
            d_val = np.mat(A[0:3, 3] - np.dot(rot_x, B[0:3, 3]))
            c_s.append(c_val)
            d_s.append(d_val)
        
        def objective(b_x):
            b_x = np.mat(b_x).T
            summation = 0
            for c, d in zip(c_s, d_s):
                val = c*b_x + d
                summation += np.linalg.norm(val)
            return summation

        b_x = np.array([0., 0., 0.])
        res = minimize(objective, b_x)
        print(res.success)
        print(res.x)
        
        X = np.eye(4)
        X[0:3, 0:3] = rot_x 
        X[0:3, 3] = res.x 
        return X

    def solve_axxb_exact(self, ABs): #TODO(oleguer): Refactor this!!
        def get_repr_vect(mat):
            return np.mat((mat[2, 1], -mat[2, 0], mat[1, 0]))
        
        # 3. Solve AX=XB problem
        M = np.zeros((3, 3))
        alphas = []
        betas = []
        for A, B in ABs:
            A = np.mat(A)
            B = np.mat(B)
            alpha_m = logm(A[0:3, 0:3])
            alpha_v = get_repr_vect(alpha_m)
            beta_m = logm(np.mat(B[0:3, 0:3]))
            beta_v = get_repr_vect(beta_m)
            betas.append(beta_v.T)
            alphas.append(alpha_v.T)
            # print(beta_v.T)
            # print(alpha_v)

        a1 = alphas[0].T
        a2 = alphas[1].T
        a3 = np.mat(np.cross(np.array(alphas[0].T), np.array(alphas[1].T)))
        print(a1)
        print(a2)
        print(a3)
        A = np.zeros((3, 3))
        A[0:3, 0] = a1
        A[0:3, 1] = a2
        A[0:3, 2] = a3
        print(A)
        print("")

        b1 = betas[0].T
        b2 = betas[1].T
        b3 = np.mat(np.cross(np.array(betas[0].T), np.array(betas[1].T)))
        print(b1)
        print(b2)
        print(b3)
        B = np.zeros((3, 3))
        B[0:3, 0] = b1
        B[0:3, 1] = b2
        B[0:3, 2] = b3
        print(B)
        print("")

        rot_x = np.mat(A)*np.mat(np.linalg.inv(B))
        print(rot_x)
        del A
        del B
        
        c_s = []
        d_s = []
        for A, B in ABs:
            A = np.mat(A)
            B = np.mat(B)
            c_val = np.mat(A[0:3, 0:3] - np.eye(3)) #NOTE: Swapped that for minimization
            d_val = np.mat(A[0:3, 3] - np.dot(rot_x, B[0:3, 3]))
            c_s.append(c_val)
            d_s.append(d_val)
        
        def objective(b_x):
            b_x = np.mat(b_x).T
            summation = 0
            for c, d in zip(c_s, d_s):
                val = c*b_x + d
                summation += np.linalg.norm(val)
            return summation

        b_x = np.array([0, 0, 0])
        res = minimize(objective, b_x)
        print(res.success)
        print(res.x)

        return 0

    def plot(self):
        '''Debug function to make sure corners and apriltag make sense
        '''
        img = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        cv2.circle(img, (self.apriltag_center[0], self.apriltag_center[1]), 5, (0, 0, 255), -1)
        img = cv2.drawChessboardCorners(img, self.board_shape, self.corners, self.found)
        cv2.imshow("img", img)
        cv2.waitKey(0)

    # PRIVATE
    def __rigid_transform_3D(self, A, B):
        ''' Returns transformation matrix between two sets of 3D points
        '''

        A = np.mat(A)
        B = np.mat(B)
        # assert len(A) == len(B)
        # N = A.shape[0]; # total points
        # centroid_A = np.mean(A, axis=0)
        # centroid_B = np.mean(B, axis=0)

        # # centre the points
        # AA = A - np.tile(centroid_A, (N, 1))
        # BB = B - np.tile(centroid_B, (N, 1))
        # H = np.dot(AA.T, BB)
        # U, S, Vt = np.linalg.svd(H)
        # R = np.dot(Vt.T, U.T)

        # # special reflection case
        # if np.linalg.det(R) < 0:
        #     print("Reflection detected")
        #     Vt[2,:] *= -1
        #     R = np.dot(Vt.T, U.T)
        # t = -np.dot(R,centroid_A) + centroid_B.T
        
        # TODO(oleguer): Implement it here
        R, t = rigid_transform_3D(A, B)

        M = np.eye(4)
        M[0:3, 0:3] = R
        M[0:3, 3] = t.flatten()
        return M

    def __get_oriented_corners(self, image):
        # 0. Reset debug variables: TODO(oleguer): Remove this when everything works
        self.found = False
        self.corners = []
        self.apriltag = None
        self.image = image
        # 1. Get chessboard corners in pixel position
        self.found, unoriented_corners = self.__get_corners(image, False)
        # 2. Get apriltag center
        self.apriltag_center = self.get_apriltag_center(image)
        assert(self.apriltag_center is not None)
        # 3. Orient corners
        self.corners = self.__orient_corners(unoriented_corners, self.apriltag_center)
        # self.corners = unoriented_corners #TODO(oleguer): Corners orienting doesnt work, fix it
        return self.corners  #TODO(oleguer): Shouldnt be returning self variable, fix this

    def __get_corners(self, img, subpixel = False):
        '''Given image returns list of checker corners
        '''
        # Convert to grayscale if its not
        if (len(img.shape) != 2):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        found, corners = cv2.findChessboardCorners(img, self.board_shape)

        if not found:
            print("ERROR: Corners not found!")

        if not found or not subpixel:
            return found, corners

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # termination criteria
        corners_subpx = cv2.cornerSubPix(img, corners, self.board_shape, (-1,-1), criteria)  # subpixel accuracy
        return True, corners_subpx

    def get_apriltag_center(self, img):
        options = apriltag.DetectorOptions(families=self.apriltag_families)
        detector = apriltag.Detector(options=options)
        result = detector.detect(img)
        if len(result) == 0:
            print("Apriltag not found!")
            return None
        return np.array(result[0].center, dtype=np.int16)

    # TODO:(oleguer) REVIEW THIS!!!
    def __orient_corners(self, corners, april_pos):
        '''Makes sure all corners are sorted in the following way:
        a_tag
        C1 -> C2 -> ... -> Cn
        Cn+1 -> ...     -> C2n
        ...             -> Cnn
        '''
        n = self.board_shape[1] - 1
        m = self.board_shape[0] - 1
        corners_mat = np.reshape(corners, (n+1, m+1, 1, 2))

        # Get closest corner
        distances = []
        distances.append(dist.euclidean(april_pos, corners_mat[0][0])) # 0
        distances.append(dist.euclidean(april_pos, corners_mat[n][0])) # 1
        distances.append(dist.euclidean(april_pos, corners_mat[n][m])) # 2
        distances.append(dist.euclidean(april_pos, corners_mat[0][m])) # 3
        closest_corner = np.argmin(distances)

        if closest_corner == 1:
            corners_mat = np.flip(corners_mat, axis=0)
        elif closest_corner == 2:
            corners_mat = np.flip(corners_mat, axis=0)
            corners_mat = np.flip(corners_mat, axis=1)
        elif closest_corner == 3:
            corners_mat = np.flip(corners_mat, axis=1)

        # Need to transpose?
        april_v = (april_pos[0] - corners_mat[0][0][0][0], april_pos[1] - corners_mat[0][0][0][1])
        second_v = (corners_mat[0][1][0][0] - corners_mat[0][0][0][0], corners_mat[0][1][0][1] - corners_mat[0][0][0][1])
        april_v = april_v/np.linalg.norm(april_v, ord = 2)
        second_v = second_v/np.linalg.norm(second_v, ord = 2)
        if (abs(np.dot(april_v, second_v)) > 0.5):
            print("Transposing to fix orientation")
            corners_mat.transpose()

        # Return flattened corners
        corners = corners_mat.reshape((n+1)*(m+1), 1, 2)
        return corners

if __name__ == "__main__":
    pass
