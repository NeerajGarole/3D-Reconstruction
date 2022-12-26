import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from q2_1_eightpoint import eightpoint
from q3_2_triangulate import findM2
from q4_1_epipolar_correspondence import epipolarCorrespondence

# Insert your package here


'''
Q4.2: Finding the 3D position of given points based on epipolar correspondence and triangulation
    Input:  temple_pts1, chosen points from im1
            intrinsics, the intrinsics dictionary for calling epipolarCorrespondence
            F, the fundamental matrix
            im1, the first image
            im2, the second image
    Output: P (Nx3) the recovered 3D points
    
    Hints:
    (1) Use epipolarCorrespondence to find the corresponding point for [x1 y1] (find [x2, y2])
    (2) Now you have a set of corresponding points [x1, y1] and [x2, y2], you can compute the M2
        matrix and use triangulate to find the 3D points. 
    (3) Use the function findM2 to find the 3D points P (do not recalculate fundamental matrices)
    (4) As a reference, our solution's best error is around ~2000 on the 3D points. 
'''
def compute3D_pts(temple_pts1, intrinsics, F, im1, im2):
    # ----- TODO -----
    # YOUR CODE HERE
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    temple_pts2 = np.zeros((temple_pts1.shape[0], 2))

    for i in range(temple_pts1.shape[0]):
        x, y = epipolarCorrespondence(im1, im2, F, temple_pts1[i, 0], temple_pts1[i, 1])
        temple_pts2[i, 0] = x
        temple_pts2[i, 1] = y

    M2, C2, P = findM2(F, temple_pts1, temple_pts2, intrinsics)
    M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    C1 = np.matmul(K1, M1)
    np.savez("result/q4_2.npz", F=F, M1=M1, M2=M2, C1=C1, C2=C2)
    return P



'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
if __name__ == "__main__":

    temple_coords_path = np.load('data/templeCoords.npz')
    correspondence = np.load('data/some_corresp.npz')  # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz')  # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    # ----- TODO -----
    # YOUR CODE HERE
    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    pts1 = np.hstack([temple_coords_path["x1"], temple_coords_path["y1"]])
    P = compute3D_pts(pts1, intrinsics, F, im1, im2)
    ax = plt.axes(projection="3d")
    ax.scatter3D(P[:, 0], P[:, 1], P[:, 2], color="blue")
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(90, 0)
    plt.show()