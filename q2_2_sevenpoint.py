import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, _singularize, refineF
import numpy.polynomial.polynomial as npp
# Insert your package here


'''
Q2.2: Seven Point Algorithm for calculating the fundamental matrix
    Input:  pts1, 7x2 Matrix containing the corresponding points from image1
            pts2, 7x2 Matrix containing the corresponding points from image2
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated 3x3 fundamental matrixes.
    
    HINTS:
    (1) Normalize the input pts1 and pts2 scale paramter M.
    (2) Setup the seven point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Pick the last two colum vector of vT.T (the two null space solution f1 and f2)
    (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
        det(F) = 0. Sovling this polynomial will give you one or three real solutions of the fundamental matrix. 
        Use np.polynomial.polynomial.polyroots to solve for the roots
    (6) Unscale the fundamental matrixes and return as Farray
'''
def func_sol(a, F1, F2):
    return np.linalg.det(a * F1 + (1 - a) * F2)

def sevenpoint(pts1, pts2, M):
    Farray = []
    pts1N = pts1 / M
    pts2N = pts2 / M
    x1 = pts1N[:, 0]
    y1 = pts1N[:, 1]
    x2 = pts2N[:, 0]
    y2 = pts2N[:, 1]
    one = np.ones(pts1.shape[0])
    A = np.transpose(np.vstack((x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, one)))
    u, s, v = np.linalg.svd(A)
    F1 = v[-1, :].reshape(3, 3)
    F2 = v[-2, :].reshape(3, 3)
    a0 = func_sol(0,F1,F2)
    a1 = 2 * (func_sol(1,F1,F2) - func_sol(-1,F1,F2)) / 3 - (func_sol(2 ,F1,F2) - func_sol(-2,F1,F2)) / 12
    a2 = 0.5 * func_sol(1,F1,F2) + 0.5 * func_sol(-1,F1,F2) - a0
    a3 = func_sol(1,F1,F2) - a0 - a1 - a2
    roots = np.roots(np.array([a3, a2, a1, a0]))
    T = np.array([[1 / M, 0, 0], [0, 1 / M, 0], [0, 0, 1]])
    for root in roots:
        Ftmp = root * F1 + (1 - root) * F2
        Ftmp = refineF(Ftmp, pts1 / M, pts2 / M)
        Ftmp = np.dot((np.dot(np.transpose(T), Ftmp)), T)
        Farray.append(Ftmp)
    return Farray


if __name__ == "__main__":
        
    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    # indices = np.arange(pts1.shape[0])
    # indices = np.random.choice(indices, 7, False)
    indices = np.array([82, 19, 56, 84, 54, 24, 18])

    M = np.max([*im1.shape, *im2.shape])

    Farray = sevenpoint(pts1[indices, :], pts2[indices, :], M)

    print(Farray[2])

    F = Farray[2]
    F = F/F[2,2]
    print(F)
    np.savez('result/q2_2.npz', F, M, pts1, pts2)

    # fundamental matrix must have rank 2!
    assert(np.linalg.matrix_rank(F) == 2)
    # displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution. 
    np.random.seed(1) #Added for testing, can be commented out

    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    max_iter = 500
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    ress = []
    F_res = []
    choices = []
    M=np.max([*im1.shape, *im2.shape])

    for i in range(max_iter):
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        Fs = sevenpoint(pts1_choice, pts2_choice, M)

        for F in Fs:
            choices.append(choice)
            res = calc_epi_error(pts1_homo,pts2_homo, F)
            F_res.append(F)
            ress.append(np.mean(res))
            
    min_idx = np.argmin(np.abs(np.array(ress)))
    F = F_res[min_idx]
    print("Error:", ress[min_idx])
    F = F/F[2,2]
    print("F = ")
    print(F)
    displayEpipolarF(im1, im2, F)

    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)