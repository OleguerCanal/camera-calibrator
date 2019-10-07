import apriltag
import cv2
import numpy as np
from scipy.spatial import distance as dist
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm, sqrtm, inv
import yaml

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
        M[3, 0:3] = translation
        return M

    #TODO(oleguer): Return intrinsics as well
    def get_extrinsics(self, image):
        '''Given checkerboard+apriltag image, returns extrinsics:
        transformation matrix from checkerboard to camera frame
        '''
        # 0. Reset debug variables: TODO(oleguer): Remove this when everything works
        self.found = False
        self.corners = []
        self.apriltag = None
        self.image = image

        # 1. Get checkerbard corners in pixel position
        self.found, unoriented_corners = self.__get_corners(image)

        # 2. Get apriltag center
        self.apriltag_center = self.__get_apriltag_center(image)

        # 3. Orient corners
        self.corners = self.__orient_corners(unoriented_corners, self.apriltag_center)

        # 4. Get matrix of real positions
        corners_m = np.zeros((self.boardSize[0], self.boardSize[1], 3), np.float32)
        for i in range(self.boardSize[0]):
            for j in range(self.boardSize[1]):
                corners_m[i][j] = (i*self.tileSide, j*self.tileSide, 0)

        # 5. Compute transformation
        ret, intrinsics_mat, distortion_coef, rotation_vect, translation_vect =\
            cv2.calibrateCamera(corners_m, corners_px, image_shape, None, None)
        rot_matrix = R.from_rotvec(rotation_vect).as_dcm()
        transformation_matrix = np.eye(4)
        transformation_matrix[0:3,0:3] = rot_matrix
        transformation_matrix[3,0:3] = translation_vect #TODO(oleguer): Maybe transpose this
        return transformation_matrix

    def eye_in_hand_finetunning(self, transforms, images):
        ''' Given a list of rough transformations (Ai) and a list of corresponding checkerboard images
            returns transformation correction from Ai to real camera position
        '''
        # 1. Compute all B_i matrices
        B_is = []
        for image in images: 
            B_i = self.get_extrinsics(image)
            B_is.append(B_i)

        # 2. Combine all A_is, B_is to create all possible A, B
        zipped_ABs = []
        for i in range(0, len(transforms)):
            for j in range(i+1, len(transforms)):
                A_i = transforms[i]
                A_j = transforms[j]
                A = np.dot(A_i, np.inverse(A_j))  # TODO(Oleguer): Review this
                B_i = B_is[i]
                B_j = B_is[j]
                B = np.dot(B_i, np.inverse(B_j))  # TODO(Oleguer): Review this
                zipped_ABs.append((A, B))

        # 3. Solve AX=XB problem
        M = np.zeros((3, 3))
        for A, B in zipped_ABs:
            alpha = logm(A[0:3, 0:3])
            beta = logm(B[0:3, 0:3])
            M += beta*np.transpose(alpha)

        rot_x = np.dot(inv(sqrtm(np.dot(np.transpose(M), M))), np.transpose(M))

        c = []
        d = []
        for A, B in data:
            c_val = np.eye(3) - A[0:3, 0:3]
            d_val = A[3, 0:3] - np.dot(rot_x, B[3, 0:3])
            c.append(c_val)
            d.append(d_val)

        trans_x = np.dot(np.dot(inv(np.dot(np.transpose(c), c)), np.transpose(c)), d)
        
        X = np.eye(4)
        X[0:3, 0:3] = rot_x 
        X[3, 0:3] = trans_x 
        return X

    def plot(self):
        '''Debug function to make sure corners and apriltag make sense
        '''
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.circle(self.img, (self.apriltag_center[0], self.apriltag_center[1]), 5, (0, 0, 255), -1)
        img = cv2.drawChessboardCorners(self.image, self.boardSize, self.corners, self.found)
        cv2.imshow("img", img)
        cv2.waitKey(0)

    # PRIVATE
    def __get_corners(self, img, subpixel = False):
        '''Given image returns list of checker corners
        '''
        # Convert to grayscale if its not
        if (len(img.shape) != 2):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        found, corners = cv2.findChessboardCorners(img, self.boardSize)

        if found:
            print("Found corners!")
        else:
            print("ERROR: Corners not found!")

        if not found or not subpixel:
            return found, corners

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # termination criteria
        corners_subpx = cv2.cornerSubPix(img, corners, self.boardSize, (-1,-1), criteria)  # subpixel accuracy
        return True, corners_subpx

    def __get_apriltag_center(self, img):
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
        matrix_corners = np.reshape(corners, (self.boardSize[0], self.boardSize[1], 1, 2))

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
    # Load sample transforms
    with open('data/transforms.yaml') as f:
        transforms_data = yaml.load(f, Loader=yaml.FullLoader)
        print(data["exp" + str(1)]["Translation"])
        print(data["exp" + str(1)]["Quaternion"])
    
    # Instantiate class
    calib = CameraCalibrator(board_shape=(3, 4), tile_side=0.062, apriltag_families="tag36h10")

    transforms = []
    images = []
    for i in range(1, 7):
        image = cv2.imread("data/" + str(i) + ".png", cv2.IMREAD_GRAYSCALE)
        trans = transforms_data["exp" + str(i)]["Translation"]
        quat = transforms_data["exp" + str(i)]["Quaternion"]
        transforms_data.append(calib.transquat_to_mat(trans, quat))
        images.append(image)

    X = calib.eye_in_hand_finetunning(transforms, images)
    