import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF, _singularize

# Insert your package here



'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T.
    (2) Setup the eight point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. 
        (Remember to usethe normalized points instead of the original points)
    (6) Unscale the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    pts1N = pts1 / M
    pts2N = pts2 / M

    x1 = pts1N[:, 0]
    y1 = pts1N[:, 1]
    x2 = pts2N[:, 0]
    y2 = pts2N[:, 1]
    one = np.ones(pts1.shape[0])

    A = np.transpose(np.vstack((x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, one)))
    u, s, v = np.linalg.svd(A)
    F = refineF(v[-1].reshape(3, 3), pts1N, pts2N)
    # F = _singularize(F) # Singularity enforced when refineF returns F
    T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    F = np.dot((np.dot(np.transpose(T), F)), T)
    return F




if __name__ == "__main__":
        
    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    F = F/F[2, 2]
    print(F)
    # Save F:
    np.savez('result/q2_1.npz', F=F, M=np.max([*im1.shape, *im2.shape]))
    # q2_1 = np.load('result/q2_1.npz')
    # F = q2_1['F']
    # M = q2_1['M']

    # Q2.1
    displayEpipolarF(im1, im2, F)



    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)