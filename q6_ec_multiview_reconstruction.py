import numpy as np
import matplotlib.pyplot as plt

import os

from helper import visualize_keypoints, plot_3d_keypoint, connections_3d, colors
from q3_2_triangulate import triangulate

# Insert your package here

'''
Q6.1 Multi-View Reconstruction of keypoints.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx3 matrix with the 2D image coordinates and confidence per row
            C2, the 3x4 camera matrix
            pts2, the Nx3 matrix with the 2D image coordinates and confidence per row
            C3, the 3x4 camera matrix
            pts3, the Nx3 matrix with the 2D image coordinates and confidence per row
    Output: P, the Nx3 matrix with the corresponding 3D points for each keypoint per row
            err, the reprojection error.
'''
def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres=100):
    P, err = triangulate(C2, pts2[:, :2], C3, pts3[:, :2])
    return P, err


'''
Q6.2 Plot Spatio-temporal (3D) keypoints
    :param car_points: np.array points * 3
'''
def plot_3d_keypoint_video(pts_3d_video):
    # Replace pass by your implementation
    ax = plt.axes(projection='3d')
    N = pts_3d_video.shape[0]
    for i in range(N):
        for j in range(len(connections_3d)):
            x = [pts_3d_video[i, connections_3d[j][0], 0], pts_3d_video[i, connections_3d[j][1], 0]]
            y = [pts_3d_video[i, connections_3d[j][0], 1], pts_3d_video[i, connections_3d[j][1], 1]]
            z = [pts_3d_video[i, connections_3d[j][0], 2], pts_3d_video[i, connections_3d[j][1], 2]]
            ax.plot(x, y, z, color=colors[j])
    plt.show()


#Extra Credit
if __name__ == "__main__":

    err_lst = []
    pts_3d_video = []
    P_lst = []
    for loop in range(10):
        print(f"processing time frame - {loop}")

        data_path = os.path.join('data/q6/','time'+str(loop)+'.npz')
        image1_path = os.path.join('data/q6/','cam1_time'+str(loop)+'.jpg')
        image2_path = os.path.join('data/q6/','cam2_time'+str(loop)+'.jpg')
        image3_path = os.path.join('data/q6/','cam3_time'+str(loop)+'.jpg')

        im1 = plt.imread(image1_path)
        im2 = plt.imread(image2_path)
        im3 = plt.imread(image3_path)

        data = np.load(data_path)
        pts1 = data['pts1']
        pts2 = data['pts2']
        pts3 = data['pts3']

        K1 = data['K1']
        K2 = data['K2']
        K3 = data['K3']

        M1 = data['M1']
        M2 = data['M2']
        M3 = data['M3']

        C1 = np.matmul(K1,M1)
        C2 = np.matmul(K2,M2)
        C3 = np.matmul(K3,M3)

        #Note - Press 'Escape' key to exit img preview and loop further
        # img = visualize_keypoints(im2, pts2)


        # YOUR CODE HERE
        P, e = MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3)
        P_lst.append(P)
        err_lst.append(e)

np.savez("result/q6_1.npz", P=P_lst[0])
plot_3d_keypoint(P_lst[0]) # least error P

plot_3d_keypoint_video(np.asarray(P_lst))

