import numpy as np
import matplotlib.pyplot as plt

from helper import camera2
from q2_1_eightpoint import eightpoint
from q3_1_essential_matrix import essentialMatrix

# Insert your package here


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.

    Hints:
    (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
    (2) Solve for the least square solution using np.linalg.svd
    (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from 
        homogeneous coordinates to non-homogeneous ones)
    (4) Keep track of the 3D points and projection error, and continue to next point 
    (5) You do not need to follow the exact procedure above. 
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    P = np.zeros((pts1.shape[0], 3))
    err = 0

    for i in range(pts1.shape[0]):
        x1 = pts1[i, 0]
        y1 = pts1[i, 1]
        x2 = pts2[i, 0]
        y2 = pts2[i, 1]
        A = np.asarray([x1 * C1[2, :] - C1[0, :], y1 * C1[2, :] - C1[1, :],
                        x2 * C2[2, :] - C2[0, :], y2 * C2[2, :] - C2[1, :]])
        u, s, v = np.linalg.svd(A)
        p = np.transpose(v)[:, -1]
        p = p / p[-1]
        P[i, :] = p[:3]
        p1 = np.matmul(C1, p)
        p2 = np.matmul(C2, p)
        e1 = np.sum((p1[:2]/p1[-1] - pts1[i, :]) ** 2)
        e2 = np.sum((p2[:2]/p2[-1] - pts2[i, :]) ** 2)
        err = err + e1 + e2

    return P, err

'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

def findM2(F, pts1, pts2, intrinsics, filename = 'q3_3.npz'):
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']
    E = essentialMatrix(F, K1, K2)
    M1 = np.concatenate((np.eye(3), np.zeros((3, 1))), axis = 1)
    C1 = np.matmul(K1, M1)
    M2_s = camera2(E)

    for i in range(4):
        M2 = M2_s[:, :, i]
        C2 = np.matmul(K2, M2)
        P, err = triangulate(C1, pts1, C2, pts2)
        if np.min(P[:, -1]) > 0:
            break

    return M2, C2, P



if __name__ == "__main__":

    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    M2, C2, P = findM2(F, pts1, pts2, intrinsics)
    np.savez("result/q3_3.npz", M2=M2, C2=C2, P=P)
    # Simple Tests to verify your implementation:
    M1 = np.hstack((np.identity(3), np.zeros(3)[:,np.newaxis]))
    C1 = K1.dot(M1)
    C2 = K2.dot(M2)
    P_test, err = triangulate(C1, pts1, C2, pts2)
    print(err)
    assert(err < 500)