import apriltag
import cv2
import numpy as np
from scipy.spatial import distance as dist
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm, sqrtm, inv, expm
from scipy.optimize import minimize
from math import sqrt
import yaml
from utils.rigid_trans_test import rigid_transform_3D  #TODO(oleguer): Remove that

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
    def get_extrinsics_2D(self, image):
        '''Given checkerboard+apriltag 2D image, returns extrinsics:
        transformation matrix from checkerboard to camera frame
        '''
        # 0. Reset debug variables: TODO(oleguer): Remove this when everything works
        self.found = False
        self.corners = []
        self.apriltag = None
        self.image = image

        # 1. Get chessboard corners in pixel position
        self.found, unoriented_corners = self.__get_corners(image)

        # 2. Get apriltag center
        self.apriltag_center = self.get_apriltag_center(image)

        # 3. Orient corners
        self.corners = self.__orient_corners(unoriented_corners, self.apriltag_center)

        # 4. Get matrix of chessboard frame values
        corners_m = []
        for i in range(self.board_shape[1]): #TODO(oleguer): Review board_shape params!
            for j in range(self.board_shape[0]):
                corners_m.append([i*self.tile_side, j*self.tile_side, 0])
        corners_m = [np.array(corners_m, dtype=np.float32)]

        # print(corners_m)
        # print(self.corners)
        # self.plot()

        # 5. Compute transformation
        ret, intrinsics_mat, distortion_coef, rotation_vect, translation_vect =\
            cv2.calibrateCamera(corners_m, [self.corners], self.image.shape[::-1], None, None)
        rotation_vect = np.array(rotation_vect).reshape(3)
        translation_vect = np.array(translation_vect).reshape(3)

        # From camera to checkerboard
        rot_matrix = R.from_rotvec(rotation_vect).as_dcm()
        camera_to_chessboard = np.eye(4)
        camera_to_chessboard[0:3,0:3] = rot_matrix
        camera_to_chessboard[0:3, 3] = translation_vect
        
        # From checkerboard to camera
        chessboard_to_camera = np.eye(4)
        chessboard_to_camera[0:3,0:3] = rot_matrix.T
        chessboard_to_camera[0:3, 3] = -translation_vect
        return chessboard_to_camera

    def get_extrinsics_3D(self, image, xyz_coordinates_matrix):
        ''' Given image of a checkerboard with apriltag and xyz_coordinates_matrix,
            Returns transformation matrix from checkerboard to camera frame
            
            xyz_coordinates_matrix meaning: a 3 channel numpy array where:
            channel 0: x coordinates
            channel 1: y coordinates
            channel 3: z coordinates
            NOTE: Make sure the channels are in the rigth order!!!! 

            (Use this if you have a 3D sensor, otherwise use get_extrinsics_2D)
        '''
        # 0. Reset debug variables: TODO(oleguer): Remove this when everything works
        self.found = False
        self.corners = []
        self.apriltag = None
        self.image = image

        # 1. Get checkerbard corners in pixel position
        self.found, unoriented_corners = self.__get_corners(image)

        # 2. Get apriltag center
        self.apriltag_center = self.get_apriltag_center(image)

        # 3. Orient corners
        self.corners = self.__orient_corners(unoriented_corners, self.apriltag_center)

        # 4.Get matrix of camera frame values
        corners_camera = []
        for corner in self.corners:
            corner = corner[0]
            x = xyz_coordinates_matrix[int(corner[1])][int(corner[0])][0]
            y = xyz_coordinates_matrix[int(corner[1])][int(corner[0])][1]
            z = xyz_coordinates_matrix[int(corner[1])][int(corner[0])][2]
            corner_camera = [x, y, z]
            corners_camera.append(corner_camera)

        # 5. Get matrix of chessboard frame values
        corners_chessboard = []
        for i in range(self.board_shape[1]): #TODO(oleguer): Review board_shape params!
            for j in range(self.board_shape[0]):
                corners_chessboard.append([i*self.tile_side, j*self.tile_side, 0])

        corners_camera = np.array(corners_camera)
        corners_chessboard = np.array(corners_chessboard)
        print(corners_camera.shape)
        print(corners_chessboard.shape)
        chessboard_to_camera = self.__rigid_transform_3D(corners_chessboard, corners_camera)
        return chessboard_to_camera

    def eye_in_hand_finetunning(self, transforms, images):
        ''' Given a list of rough transformations (Ai) and a list of corresponding checkerboard images
            returns transformation correction from Ai to real camera position
        '''
        # 1. Compute all B_i matrices
        B_is = []
        for image in images: 
            B_i = self.get_extrinsics(image)
            print(B_i)
            B_is.append(B_i)
        # a = raw_input()
        

        # 2. Combine all A_is, B_is to create all possible A, B
        zipped_ABs = []
        for i in range(0, len(transforms)):
            for j in range(i+1, len(transforms)):
                A_i = transforms[i]
                A_j = transforms[j]
                A = np.dot(np.linalg.inv(A_i), A_j)  # TODO(Oleguer): Review this
                B_i = B_is[i]
                B_j = B_is[j]
                B = np.dot(np.linalg.inv(B_i), B_j)  # TODO(Oleguer): Review this
                zipped_ABs.append((A, B))

        # 3. Solve AX=XB problem
        M = np.zeros((3, 3))
        for A, B in zipped_ABs:
            alpha = logm(A[0:3, 0:3])
            beta = logm(B[0:3, 0:3])
            M += beta*alpha.T
        rot_x = np.dot(inv(sqrtm(np.dot(M.T, M))), M.T)  # Theta_x
        print("M: ")
        print(M)
        print("rot_x: ")
        print(rot_x)

        #TODO(oleguer): Take a look at this we shouldnt be taking average but the multiplication commented down below
        trans_x = np.zeros((3, 1), dtype=np.float32)
        for A, B in zipped_ABs:
            c_val = np.eye(3) - A[0:3, 0:3]
            print("A:")
            print(A)
            print("dot:")
            print(np.dot(rot_x, B[0:3, 3]))
            d_val = A[0:3, 3] - np.dot(rot_x, B[0:3, 3])
            print(c_val)
            print(d_val)
            print(np.dot(np.linalg.inv(c_val), d_val))
            trans_x += np.dot(np.linalg.inv(c_val), d_val).T
        trans_x = trans_x/len(zipped_ABs)

        #OLD CODE WHICH SHOULD BE BETTER
        # c = []
        # d = []
        # for A, B in zipped_ABs:
        #     c_val = np.eye(3) - A[0:3, 0:3]
        #     d_val = A[0:3, 3] - np.dot(rot_x, B[0:3, 3])
        #     c.append(c_val)
        #     d.append(d_val)
            
        
        # c = np.array(c, dtype=np.float32)
        # d = np.array(d, dtype=np.float32)
        # print(c.shape)
        # print(d.shape)
        # print(np.dot(c, c).shape)
        # trans_x = np.dot(np.dot(np.linalg.inv(np.dot(c.T, c)), c.T), d)
        
        X = np.eye(4)
        X[0:3, 0:3] = rot_x 
        X[0:3, 3] = trans_x 
        return X

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

    def __get_corners(self, img, subpixel = False):
        '''Given image returns list of checker corners
        '''
        # Convert to grayscale if its not
        if (len(img.shape) != 2):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        found, corners = cv2.findChessboardCorners(img, self.board_shape)

        if found:
            print("Found corners!")
        else:
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
        if len(result) > 0:
            print("Found apriltag!")
        else:
            return None
        return np.array(result[0].center, dtype=np.int16)

    # TODO:(oleguer) REVIEW THIS!!!
    # TODO:(oleguer) THIS WITH SQUARE!!!
    def __orient_corners(self, corners, april_pos):
        '''Makes sure all corners are sorted in the following way:
        a_tag
        C1 -> C2 -> ... -> Cn
        Cn+1 -> ...     -> C2n
        ...             -> Cnn
        '''
        corners_mat = np.reshape(corners, (self.board_shape[0], self.board_shape[1], 1, 2))

        # Get closest corner
        n = corners_mat.shape[0] - 1
        m = corners_mat.shape[1] - 1
        distances = []
        distances.append(dist.euclidean(april_pos, corners_mat[0][0])) # 0
        distances.append(dist.euclidean(april_pos, corners_mat[0][m])) # 1
        distances.append(dist.euclidean(april_pos, corners_mat[n][m])) # 2
        distances.append(dist.euclidean(april_pos, corners_mat[n][0])) # 3
        closest_corner = np.argmin(distances)

        # print(corners_mat)
        if closest_corner == 1:
            corners_mat = np.rot90(corners_mat, 2).T
        elif closest_corner == 2:
            corners_mat = np.rot90(corners_mat)
            corners_mat = np.rot90(corners_mat)
        elif closest_corner == 3:
            corners_mat = np.rot90(corners_mat, 2).T
            corners_mat = np.transpose(corners_mat)
        # print(corners_mat)

        # Need to transpose?
        april_v = (april_pos[0] - corners_mat[0][0][0][0], april_pos[1] - corners_mat[0][0][0][1])
        second_v = (corners_mat[0][1][0][0] - corners_mat[0][0][0][0], corners_mat[0][1][0][1] - corners_mat[0][0][0][1])
        april_v = april_v/np.linalg.norm(april_v, ord = 2)
        second_v = second_v/np.linalg.norm(second_v, ord = 2)
        if (abs(np.dot(april_v, second_v)) > 0.5):
            print("Transposing to correct orientation")
            corners_mat.transpose()

        # Return flattened corners
        corners = corners_mat.reshape((n+1)*(m+1), 1, 2)
        return corners

if __name__ == "__main__":
    pass