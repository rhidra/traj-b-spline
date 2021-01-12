import autograd.numpy as np, math

"""
Utility functions to manipulate B-splines
"""

M_6 = (1 / math.factorial(5)) * np.array([[1,26,66,26,1,0], [-5,-50,0,50,5,0], [10,20,-60,20,10,0], [-10,20,0,-20,10,0], [5,-20,30,-20,5,0], [-1,5,-10,10,-5,1]])

U_1 = lambda u: np.array([1, u, u*u, u*u*u, u*u*u*u, u*u*u*u*u]).reshape(1,6)
U_2 = lambda u: np.array([0, 1, 2*u, 3*u*u, 4*u*u*u, 5*u*u*u*u]).reshape(1,6)
U_3 = lambda u: np.array([0, 0, 2, 6*u, 12*u*u, 20*u*u*u]).reshape(1,6)
U_4 = lambda u: np.array([0, 0, 0, 6, 24*u, 60*u*u])
U_5 = lambda u: np.array([0, 0, 0, 0, 24, 120*u])

bspline = lambda u, pts: U_1(u).dot(M_6).dot(pts).reshape(-1)
bsplineVel = lambda u, pts, delta_t: (1/delta_t) * np.dot(np.dot(U_2(u), M_6), pts).reshape(-1)
bsplineAcc = lambda u, pts, delta_t: (1/delta_t/delta_t) * np.dot(np.dot(U_3(u), M_6), pts).reshape(-1)
bsplineJerk = lambda u, pts, delta_t: (1/delta_t/delta_t/delta_t) * np.dot(np.dot(U_4(u), M_6), pts).reshape(-1)
bsplineSnap = lambda u, pts, delta_t: (1/delta_t/delta_t/delta_t/delta_t) * np.dot(np.dot(U_5(u), M_6), pts).reshape(-1)

gradPts = lambda i: np.array([0]*i + [1] + [0]*(6*2-i-1)).reshape(6, 2)
gradBspline = lambda i, u: U_1(u).dot(M_6).dot(gradPts(i))
gradBsplineVel = lambda i, u, delta_t: U_2(u).dot(M_6).dot(gradPts(i)) /delta_t
gradBsplineAcc = lambda i, u, delta_t: U_3(u).dot(M_6).dot(gradPts(i)) /delta_t/delta_t
gradBsplineJerk = lambda i, u, delta_t: U_4(u).dot(M_6).dot(gradPts(i)) /delta_t/delta_t/delta_t
gradBsplineSnap = lambda i, u, delta_t: U_5(u).dot(M_6).dot(gradPts(i)) /delta_t/delta_t/delta_t/delta_t


# Generate a smooth path from a serie of control points
def bsplineUpsample(path, upsampleRate=5):
    smoothed = np.zeros((len(path) * upsampleRate, 2))

    for i in range(len(path)):
        p = np.array([path[np.clip(i + j - 2, 0, len(path)-1)] for j in range(6)]).reshape(6, 2)

        for uidx, u in enumerate(np.linspace(0, 1, upsampleRate)):
            smoothed[i * upsampleRate + uidx, :] = bspline(u, p)

    return smoothed


# Extract 6 points centered around idx according to the P_i = [pi-2, pi-1, pi, pi+1, pi+2, pi+3] vector from the paper
def extractPts(pts, idx):
    return pts[np.int(idx) - 2:np.int(idx) + 4]