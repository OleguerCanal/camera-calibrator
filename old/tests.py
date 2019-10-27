import numpy as np
from scipy.linalg import logm, sqrtm, inv, expm
import math
from scipy.spatial.transform import Rotation as R

def get_repr_vect(mat):
    return np.array((mat[2, 1], -mat[2, 0], mat[1, 0]))

if __name__ == "__main__":
    A1 = [[-0.989992, -0.141120, 0.000000, 0],
        [0.141120, -0.989992, 0.000000, 0],
        [0.000000, 0.000000, 1.000000, 0],
        [0, 0, 0, 1]]

    A1 = np.mat(A1)
    A1 = A1[0:3, 0:3]
    lo = logm(A1)
    lo = np.mat(lo)
    print("auto log:")
    print(np.round(lo, 2))
    # print(np.linalg.eig(lo))
    # vals, vect = np.linalg.eig(A1)
    # for val, vect in zip(vals, vect):
    #     print("VAL:", val)
    #     for v in np.array(vect):
    #         for j in v:
    #             print(j)
    #     print("####")
    print(lo.shape)
    for a in np.array(lo):
        print(a)
    alpha = get_repr_vect(lo)
    print(alpha)

    # fi = math.acos((A1.trace()-1)/2)
    # print(fi)
    # if fi > math.acos(-1):
    #     fi = math.acos(-1) - fi
    
    # hand_log = (A1-A1.T)*fi/(2*math.sin(fi))
    # print("hand log:")
    # print(hand_log)

    # a = np.mat([0, 0, 3])

    # print(a.T*a)


    # r = R.from_dcm(A1)
    # v = r.as_rotvec()
    # print(logm(v))
    # print(v)