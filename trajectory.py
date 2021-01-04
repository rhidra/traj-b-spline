import math, sys, time, matplotlib.pyplot as plt, plot, nlopt
from autograd import numpy as np, grad as autograd_grad
from phi_star import main as phi_star_gen
from utils import dist, over_sampling, Node
from config import *

M_6 = (1 / math.factorial(5)) * np.array([[1,26,66,26,1,0], [-5,-50,0,50,5,0], [10,20,-60,20,10,0], [-10,20,0,-20,10,0], [5,-20,30,-20,5,0], [-1,5,-10,10,-5,1]])

# For the quadratic cost computation (E_q)
Q_2 = lambda dt: np.array([[0]*6,[0]*6, [0,0,4,6,8,10], [0,0,6,12,18,24], [0,0,8,18,28.8,40], [0,0,10,24,40,400/7]]) / (dt*dt*dt)
Q_3 = lambda dt: np.array([[0]*6,[0]*6,[0]*6, [0,0,0,36,72,120], [0,0,0,72,192,360], [0,0,0,120,360,720]]) / (dt*dt*dt*dt*dt)
Q_4 = lambda dt: np.array([[0]*6,[0]*6,[0]*6,[0]*6, [0,0,0,0,576,1440], [0,0,0,0,1440,4800]]) / (dt*dt*dt*dt*dt*dt*dt)

# NUmber of points sampled from a single segment in a B-spline
UPSAMPLE_SCALE = 5

# Number of points being optimized at the same time
OPTIMIZED_POINTS = 3

# Teta in the distance function in the paper
OBSTACLE_DISTANCE_THRESHOLD = 5

# Objective function parameters
lambda_p, lambda_v = 1, 1 # Endpoint cost position and velocity weights
lambda_c = 1 # Collision cost weight

U_1 = lambda u: np.array([1, u, u*u, u*u*u, u*u*u*u, u*u*u*u*u]).reshape(1,6)
U_2 = lambda u: np.array([0, 1, 2*u, 3*u*u, 4*u*u*u, 5*u*u*u*u]).reshape(1,6)
U_3 = lambda u: np.array([0, 0, 2, 6*u, 12*u*u, 20*u*u*u]).reshape(1,6)
U_4 = lambda u: np.array([0, 0, 0, 6, 24*u, 60*u*u])
U_5 = lambda u: np.array([0, 0, 0, 0, 24, 120*u])

bspline = lambda u, pts: U_1(u).dot(M_6).dot(pts).reshape(-1)
bsplineVel = lambda u, pts, delta_t: (1/delta_t) * U_2(u).dot(M_6).dot(pts).reshape(-1)
bsplineAcc = lambda u, pts, delta_t: (1/delta_t/delta_t) * U_3(u).dot(M_6).dot(pts).reshape(-1)
bsplineJerk = lambda u, pts, delta_t: (1/delta_t/delta_t/delta_t) * U_4(u).reshape(1,6).dot(M_6).dot(pts).reshape(-1)
bsplineSnap = lambda u, pts, delta_t: (1/delta_t/delta_t/delta_t/delta_t) * U_5(u).reshape(1,6).dot(M_6).dot(pts).reshape(-1)

gradPts = lambda i: np.array([0]*i + [1] + [0]*(6*2-i-1)).reshape(6, 2)
gradBspline = lambda i, u: U_1(u).dot(M_6).dot(gradPts(i))
gradBsplineVel = lambda i, u, delta_t: U_2(u).dot(M_6).dot(gradPts(i)) /delta_t
gradBsplineAcc = lambda i, u, delta_t: U_3(u).dot(M_6).dot(gradPts(i)) /delta_t/delta_t
gradBsplineJerk = lambda i, u, delta_t: U_4(u).dot(M_6).dot(gradPts(i)) /delta_t/delta_t/delta_t
gradBsplineSnap = lambda i, u, delta_t: U_5(u).dot(M_6).dot(gradPts(i)) /delta_t/delta_t/delta_t/delta_t

# Extract 6 points centered around idx according to the P_i = [pi-2, pi-1, pi, pi+1, pi+2, pi+3] vector from the paper
def extractPts(pts, idx):
    assert pts.shape[1] == 2
    return pts[int(idx) - 2:int(idx) + 4]


def euclideanDistanceTransform(grid_obs):
    edt = np.zeros(grid_obs.shape)
    x, y = np.meshgrid(np.arange(grid_obs.shape[0]), np.arange(grid_obs.shape[1]))
    for pt in zip(x.reshape(-1), y.reshape(-1)):
        i0, i1 = math.floor(pt[0] - 1 - OBSTACLE_DISTANCE_THRESHOLD), math.floor(pt[0] + 1 + OBSTACLE_DISTANCE_THRESHOLD)
        j0, j1 = math.floor(pt[1] - 1 - OBSTACLE_DISTANCE_THRESHOLD), math.floor(pt[1] + 1 + OBSTACLE_DISTANCE_THRESHOLD)
        score = np.inf

        for i, row in enumerate(grid_obs[i0:i1]):
            for j, val in enumerate(row[j0:j1]):
                if val == Node.OBSTACLE:
                    d = dist(pt, np.array([i, j]))
                    if d < score:
                        score = d
        edt[pt] = score
    return edt


# Cost function for a quintic B-spline starting at startIdx
# pts and globalPts should contain all the points from the optimized trajectory and global path
def cost(pts, globalPts, startIdx, distObs, delta_t):
    # Endpoint cost
    posDiff = bspline(0, extractPts(pts, startIdx + 5)) - globalPts[5]
    velDiff = bsplineVel(0, extractPts(pts, startIdx + 5), delta_t) - np.array([1, 1]) # Hardcoded expected velocity
    E_ep = lambda_p * posDiff.dot(posDiff) + lambda_v * velDiff.dot(velDiff)

    # Collision cost
    u = np.linspace(0, 1, 1000)
    samples = np.vstack((np.repeat(np.arange(6), len(u)), np.tile(u, 6)))
    computeDist = lambda sample: distObs[tuple(bspline(sample[1], extractPts(pts, sample[0] + startIdx)).astype(np.int))]
    distances = np.apply_along_axis(computeDist, 0, samples)
    mask = distances <= OBSTACLE_DISTANCE_THRESHOLD
    distances[mask] = np.square(distances[mask] - OBSTACLE_DISTANCE_THRESHOLD) / (2 * OBSTACLE_DISTANCE_THRESHOLD)
    distances[np.invert(mask)] = 0
    velocities = np.apply_along_axis(lambda sample: np.linalg.norm(bsplineVel(sample[1], extractPts(pts, sample[0] + startIdx), delta_t)), 0, samples)
    E_c = lambda_c * np.sum(distances.dot(velocities)) / (len(u) * 6)

    # Squared derivative cost
    lambda_q2, lambda_q3, lambda_q4 = 1, 1, 1
    q2, q3, q4 = Q_2(delta_t), Q_3(delta_t), Q_4(delta_t)
    E_q = 0
    for i in range(6):
        A = M_6.dot(extractPts(pts, startIdx + i))
        B = A.T
        E_q += np.sum(lambda_q2 * B.dot(q2).dot(A) + lambda_q3 * B.dot(q3).dot(A) + lambda_q4 * B.dot(q4).dot(A))
    
    # Derivative limit cost
    max_vel, max_acc, max_jerk, max_snap = np.array([1000, 1000]), np.array([1000, 1000]), np.array([1000, 1000]), np.array([10000, 10000])
    u = np.linspace(0, 1, 10)
    samples = np.vstack((np.repeat(np.arange(6), len(u)), np.tile(u, 6)))
    def derivativeCost(pFunc, max_p, delta_t):
        def f(sample):
            p = pFunc(sample[1], extractPts(pts, sample[0] + startIdx), delta_t)
            return np.exp(p.dot(p) - max_p.dot(max_p)) - 1 if np.linalg.norm(p) > np.linalg.norm(max_p) else 0
        return f

    E_l = np.sum(np.apply_along_axis(derivativeCost(bsplineVel, max_vel, delta_t), 0, samples))
    E_l += np.sum(np.apply_along_axis(derivativeCost(bsplineAcc, max_acc, delta_t), 0, samples))
    E_l += np.sum(np.apply_along_axis(derivativeCost(bsplineJerk, max_jerk, delta_t), 0, samples))
    E_l += np.sum(np.apply_along_axis(derivativeCost(bsplineSnap, max_snap, delta_t), 0, samples))

    # Total cost
    E = E_ep + E_c + E_q + E_l

    print('{} | {} | {} | {} => {}'.format(E_ep, E_c, E_q, E_l, E))
    return E

# Similar to cost(), but return the gradient of the objective function
# each grad[i] contains the partial derivative for each (x, y) coordinate of each optimized point
def gradCost(pts, globalPts, startIdx, distObs, delta_t):
    # Endpoint cost gradient
    posDiff = bspline(0, extractPts(pts, startIdx + 5)) - globalPts[5]
    velDiff = bsplineVel(0, extractPts(pts, startIdx + 5), delta_t) - np.array([1, 1]) # Hardcoded expected velocity
    dE_ep = np.zeros(OPTIMIZED_POINTS * 2)
    for i, pti in enumerate(np.arange(6 - OPTIMIZED_POINTS, 6, .5) * 2):
        dE_ep[i] = 2 * lambda_p * posDiff.dot(gradBspline(pti, 1)) + 2 * lambda_v * velDiff.dot(gradBsplineVel(pti, 1, delta_t))

    # Collision cost gradient
    u = np.linspace(0, 1, 10)
    samples = np.vstack((np.repeat(np.arange(6), len(u)), np.tile(u, 6)))
    computeDist = lambda sample: distObs[tuple(bspline(sample[1], extractPts(pts, sample[0] + startIdx)).astype(np.int))]
    distances = np.apply_along_axis(computeDist, 0, samples)
    mask = distances <= OBSTACLE_DISTANCE_THRESHOLD
    distances[mask] = np.square(distances[mask] - OBSTACLE_DISTANCE_THRESHOLD) / (2 * OBSTACLE_DISTANCE_THRESHOLD)
    distances[np.invert(mask)] = 0
    velocities = np.apply_along_axis(lambda sample: np.linalg.norm(bsplineVel(sample[1], extractPts(pts, sample[0] + startIdx), delta_t)), 0, samples)
    dE_c = np.zeros(OPTIMIZED_POINTS * 2)
    for i, pti in enumerate(np.arange(6 - OPTIMIZED_POINTS, 6, .5) * 2):
        gradBsplineAcc(pti, )
        # E_c[i] = lambda_c * np.sum(velocities + * distances / (2 * )) / (len(u) * 6)



def optimTrajectory(path, distObs, trajDuration):
    path = np.pad(path, ((0, 6), (0, 0)), mode='edge')
    optim = np.copy(path)
    delta_t = trajDuration / len(path)
    # For each iteration, we optimize between [startOptim, startOptim + OPTIMIZED_POINTS]
    startOptim = 6

    # NLopt configuration
    def objFun(x, grad):
        if grad.size > 0:
            # Compute grad
            pass
        
        return x

    while startOptim + OPTIMIZED_POINTS <= len(path) - 6:
        # opt = nlopt.opt(nlopt.LD_LBFGS, OPTIMIZED_POINTS * 2)
        # opt.set_min_objective(objFun)
        c = cost(optim, path, startOptim - (6 - OPTIMIZED_POINTS), distObs, delta_t)
        startOptim += 1


def bsplineUpsample(path):
    smoothed = np.zeros((len(path) * UPSAMPLE_SCALE, 2))

    for i in range(len(path)):
        p = np.array([path[np.clip(i + j - 2, 0, len(path)-1)] for j in range(6)]).reshape(6, 2)

        for uidx, u in enumerate(np.linspace(0, 1, UPSAMPLE_SCALE)):
            smoothed[i * UPSAMPLE_SCALE + uidx, :] = bspline(u, p)

    return smoothed


def main():
    path, grid_obs, start, goal = phi_star_gen()

    path = np.array(over_sampling([p.pos for p in path], max_length=1))
    path[5, :] = np.array([2, 5])
    path[13, :] = np.array([5, 17])
    path[20, :] = np.array([15, 2])

    distObs = euclideanDistanceTransform(grid_obs)

    bsplinePts = optimTrajectory(path, distObs, trajDuration=10)

    smoothed = bsplineUpsample(path)

    plot.display(start, goal, grid_obs, path, smoothed, hold=True)
    

if __name__ == '__main__':
    main()