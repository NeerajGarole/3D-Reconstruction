import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous
from q2_1_eightpoint import eightpoint
from q2_2_sevenpoint import sevenpoint
from q3_2_triangulate import findM2

import scipy

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""
def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:,0], P_before[:,1], P_before[:,2], c = 'blue')
    ax.scatter(P_after[:,0], P_after[:,1], P_after[:,2], c='red')
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
 
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=10):
    nIters = 100
    prev_inliers = 0
    for i in range(nIters):
        temp_pts = np.random.choice(pts1.shape[0], size=8, replace=False)
        F_temp = eightpoint(pts1[temp_pts, :], pts2[temp_pts, :], M)
        dist = calc_epi_error(toHomogenous(pts1), toHomogenous(pts2), F_temp)

        curr_inliers = np.zeros(dist.shape[0])
        for j in range(len(dist)):
            if dist[j] < tol:
                curr_inliers[j] = 1
            else:
                curr_inliers[j] = 0
        num_inliers = np.sum(curr_inliers)

        if (num_inliers > prev_inliers):
            inliers = curr_inliers
            F = F_temp
            prev_inliers = num_inliers
    return F, inliers.astype(bool)


'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    theta = np.linalg.norm(r)
    if theta == 0:
        k_term = r
    else:
        k_term = r / theta
    k_mat = np.array([[0, -k_term[2], k_term[1]],
                      [k_term[2], 0, -k_term[0]],
                      [-k_term[1], k_term[0], 0]])
    R = np.eye(3) + np.sin(theta) * k_mat + (1 - np.cos(theta)) * (np.matmul(k_mat, k_mat))
    return R


'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    A = (R - np.transpose(R))/2
    rho = np.array([A[2, 1], A[0, 2], A[1, 0]]).reshape(-1, 1)
    s = np.linalg.norm(rho)
    c = (np.sum(np.diag(R)) - 1)/2

    if s == 0 and c == -1:
        RI = R + np.eye(3)
        for col in range(3):
            if np.sum(RI[:, col]) != 0:
                v = RI[:, col]
                break
        v_norm = np.linalg.norm(v)
        u = v / v_norm
        r = u * np.pi
        r1, r2, r3 = r[:, 0]
        r_norm = np.linalg.norm(r)
        if r_norm == np.pi and ((r1 == 0 and r2 == 0 and r3 < 0) or (r1 == 0 and r2 < 0) or (r1 < 0)):
            r = -r
        else:
            r = r
    elif s == 0 and c == 1:
        r = np.zeros((3, 1))
    else:
        u = rho / s
        r = u * np.arctan2(s, c)
    return r


'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    P = np.vstack((np.transpose(x[:-6].reshape((p1.shape[0], 3))), np.ones((1, p1.shape[0]))))
    M2 = np.hstack((rodrigues(x[-6:-3].reshape((3,))), x[-3:].reshape((3, 1))))

    C1 = np.matmul(K1, M1)
    C2 = np.matmul(K2, M2)

    p1_est_prj = np.matmul(C1, P)
    p2_est_prj = np.matmul(C2, P)

    p1_hat = np.transpose(p1_est_prj) / np.transpose(p1_est_prj)[:, -1].reshape(-1, 1)
    p1_hat = p1_hat[:, 0:2]
    p2_hat = np.transpose(p2_est_prj) / np.transpose(p2_est_prj)[:, -1].reshape(-1, 1)
    p2_hat = p2_hat[:, 0:2]
    residuals = np.concatenate([(p1 - p1_hat).reshape([-1]), (p2 - p2_hat).reshape([-1])])
    return residuals


'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    o1 = 0
    o2 = 0
    R2i = invRodrigues(M2_init[:, 0:3]).reshape(-1)
    t2i = M2_init[:, 3].reshape(-1)
    Pi = P_init.reshape(-1)
    x_init = np.concatenate((Pi, R2i, t2i))
    def rodRes(x): return rodriguesResidual(K1, M1, p1, K2, p2, x)
    o1 = (rodRes(x_init)**2).sum()
    x = scipy.optimize.leastsq(rodRes, x_init)[0]
    o2 = (rodRes(x)**2).sum()
    P2 = x[0:-6].reshape((p1.shape[0], 3))
    r2 = x[-6: -3].reshape(3, 1)
    t2 = x[-3:].reshape(3, 1)
    R2 = rodrigues(r2)
    M2 = np.concatenate((R2, t2), axis=1)

    return M2, P2, o1, o2


if __name__ == "__main__":

    np.random.seed(1)  # Added for testing, can be commented out

    some_corresp_noisy = np.load('data/some_corresp_noisy.npz')  # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz')  # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    noisy_pts1, noisy_pts2 = some_corresp_noisy['pts1'], some_corresp_noisy['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    # F = eightpoint(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]))
    # print(F/F[2,2])
    # displayEpipolarF(im1, im2, F)

    F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]))
    print(F / F[2, 2])
    # displayEpipolarF(im1, im2, F)

    pts1_inliers = noisy_pts1[inliers.squeeze(), :]
    pts2_inliers = noisy_pts2[inliers.squeeze(), :]
    F, inliers = ransacF(pts1_inliers, pts2_inliers, M=np.max([*im1.shape, *im2.shape]))     # F re-estimated

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(noisy_pts2)
    F_save = F.copy()
    F = F_save / F_save[2,2]
    assert (F.shape == (3, 3))
    assert (F[2, 2] == 1)
    assert (np.linalg.matrix_rank(F) == 2)

    # YOUR CODE HERE
    F = F_save
    # Simple Tests to verify your implementation:
    from scipy.spatial.transform import Rotation as sRot
    rotVec = sRot.random()
    mat = rodrigues(rotVec.as_rotvec())

    assert(np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3)
    assert(np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3)

    # YOUR CODE HERE
    M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    C1 = np.matmul(K1,M1)
    M2, C2, P_bef = findM2(F, pts1_inliers, pts2_inliers, intrinsics)
    M2, P_aft, o1, o2 = bundleAdjustment(K1, M1, pts1_inliers, K2, M2, pts2_inliers, P_bef)
    print("Error Before: ", o1)
    print("Error After: ", o2)
    print(F_save / F_save[2, 2])
    plot_3D_dual(P_bef, P_aft)
