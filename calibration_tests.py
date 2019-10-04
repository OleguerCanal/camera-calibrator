import apriltag
import cv2
import numpy as np
from scipy.spatial import distance as dist
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm, sqrtm, inv
import yaml

# class CameraCalibrator():
#     def __init__(self, board_shape = (3, 4), tile_side = 0.062):
#         self.board_shape = board_shape
#         self.tile_side = tile_side

def get_corners(img, boardSize, subpixel = False):
    # Convert to grayscale if its not
    if (len(img.shape) != 2):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    found, corners = cv2.findChessboardCorners(img, boardSize)

    if found:
        print("Found corners!")
    else:
        print("ERROR: Corners not found!")

    if not found or not subpixel:
        return found, corners

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # termination criteria
    corners_subpx = cv2.cornerSubPix(img, corners, boardSize, (-1,-1), criteria)  # subpixel accuracy


    return True, corners_subpx

def get_apriltag_center(img):
    # apriltag.
    options = apriltag.DetectorOptions(families='tag36h10')
    detector = apriltag.Detector(options=options)
    result = detector.detect(img)
    if len(result) > 0:
        print("Found apriltag!")
    else:
        return None
    return np.array(result[0].center, dtype=np.int16)

def sorted_corners(img, boardSize):
    found, corners = get_corners(img, boardSize)
    apriltag_center = get_apriltag_center(img)
    return found, corners, apriltag_center

# TODO:(oleguer) REVIEW THIS!!!
# TODO:(oleguer) THIS WITH SQUARE!!!
def orient(corners_mat, april_pos):
    # Get closest corner
    n = corners_mat.shape[0] - 1
    m = corners_mat.shape[1] - 1
    distances = []
    distances.append(dist.euclidean(april_pos, corners_mat[0][0])) # 0
    distances.append(dist.euclidean(april_pos, corners_mat[0][m])) # 1
    distances.append(dist.euclidean(april_pos, corners_mat[n][m])) # 2
    distances.append(dist.euclidean(april_pos, corners_mat[n][0])) # 3
    print(distances)
    closest_corner = np.argmin(distances)
    print(closest_corner)

    # Apply needed rotations:
    # for i in range(closest_corner):
    #     corners_mat = np.rot90(corners_mat)

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

def get_transformation_matrix(boardSize, tileSide, corners_px, image_shape):

    # Get matrix of real positions
    corners_m = np.zeros((boardSize[0], boardSize[1], 3), np.float32)
    for i in range(boardSize[0]):
        for j in range(boardSize[1]):
            corners_m[i][j] = (i*tileSide, j*tileSide, 0)

    # Compute transformation
    ret, intrinsics_mat, distortion_coef, rotation_vect, translation_vect =\
        cv2.calibrateCamera(corners_m, corners_px, image_shape, None, None)

    rot_matrix = R.from_rotvec().as_dcm()
    transformation_matrix = np.eye(4)
    transformation_matrix[0:3,0:3] = rot_matrix
    transformation_matrix[3,0:3] = translation_vect # Msybe transpose this
    return transformation_matrix  


def

def eye_in_hand(data):
    '''Solves AX=XB problem using least squares approach
    '''
    M = np.zeros((3, 3))
    for A, B in data:
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

def plot(img, corners, atag, found=True):
    # Show
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.circle(img, (apriltag_center[0], apriltag_center[1]), 5, (0, 0, 255), -1)
    img = cv2.drawChessboardCorners(img, boardSize, corners, found)
    cv2.imshow("img", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    with open('data/transforms.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        print(data["exp" + str(1)]["Translation"])
        print(data["exp" + str(1)]["Rotation"])

    # for i in range(1, 7):
    # img = cv2.imread("data/" + str(1) + ".png", cv2.IMREAD_GRAYSCALE)

    

    # # img = cv2.resize(img, (214, 175), interpolation = cv2.INTER_AREA)
    # # cv2.imshow("img", img)
    # # cv2.waitKey(0)
    # boardSize = (3, 4)
    # tileSide_m = 0.062

    # found, corners, apriltag_center = sorted_corners(img, boardSize)

    # plot(img, corners, apriltag_center, found)


    # matrix_corners = np.reshape(corners, (boardSize[0], boardSize[1], 1, 2))
    # corners = orient(matrix_corners, apriltag_center)

    # plot(img, corners, apriltag_center, found)   